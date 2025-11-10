# -*- coding: utf-8 -*-
import time
import grpc
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from kafka import KafkaConsumer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import squareform
import faiss
import numpy as np
import os, gc, sys
from collections import OrderedDict, defaultdict
import csv, psutil, json, math
import pathlib, random, torch, threading
import subprocess

# ======================
# Global knobs
# ======================
DEFAULT_PREFETCH_WAIT_BUDGET = 0.1  # 선택적 대기 상한(초) 예: 10ms
DEFAULT_PREFETCH_POOL_WORKERS = 8     # 프리페치 동시성 상한
DEFAULT_WARM_FRACTION = 0.25          # head-first 부분 프리페치 비율
DEFAULT_LASTN_TO_TRIGGER = 1          # 라스트-N 트리거

# Global variables
minibatch_model = None
buffered_queries = []
buffered_vectors = []
n_clusters = 10
model_name = "all-MiniLM-L6-v2"
timestamp = time.strftime("%m%d_%H%M%S")


# ======================
# Cache
# ======================
class ClusterCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()  # {cluster_id: {"embeddings": np.array, "latency": float, "count": int}}
        self.max_size = max_size

    def _calculate_eviction_priority(self, cluster_id):
        return self.cache[cluster_id]["latency"] * self.cache[cluster_id]["count"]

    def evict_if_needed_prefetch(self, new_cluster_count):
        current_size = len(self.cache)
        if current_size + new_cluster_count > self.max_size:
            num_to_evict = (current_size + new_cluster_count) - self.max_size
            eviction_candidates = sorted(self.cache.keys(), key=self._calculate_eviction_priority)[:num_to_evict]
            for cluster_id in eviction_candidates:
                del self.cache[cluster_id]
            threading.Thread(target=gc.collect, daemon=True).start()

    def evict_if_needed(self, new_cluster_count):
        current_size = len(self.cache)
        if current_size + new_cluster_count > self.max_size:
            num_to_evict = (current_size + new_cluster_count) - self.max_size
            eviction_candidates = sorted(self.cache.keys(), key=self._calculate_eviction_priority)[:num_to_evict]
            for cluster_id in eviction_candidates:
                del self.cache[cluster_id]
            threading.Thread(target=gc.collect, daemon=True).start()
            # drop_caches()  # 필요 없다면 비활성(프리페치 이점 보존)

    def get(self, cluster_id):
        if cluster_id in self.cache:
            self.cache[cluster_id]["count"] += 1
            self.cache.move_to_end(cluster_id)
            return self.cache[cluster_id]["embeddings"]
        return None

    def put(self, cluster_id, embeddings, generation_latency):
        if cluster_id in self.cache:
            self.cache[cluster_id]["embeddings"] = embeddings
            self.cache[cluster_id]["latency"] = generation_latency
            self.cache[cluster_id]["count"] += 1
        else:
            if len(self.cache) >= self.max_size:
                eviction_target = min(self.cache, key=self._calculate_eviction_priority)
                del self.cache[eviction_target]
            self.cache[cluster_id] = {"embeddings": embeddings, "latency": generation_latency, "count": 1}
            self.cache.move_to_end(cluster_id)


# ======================
# EdgeRAG With Cache
# ======================
class EdgeRAGWithCache:
    def __init__(self, ivf_centroids_path, cache_size, cluster_embedding_path, result_path,
                 prefetch_wait_budget=DEFAULT_PREFETCH_WAIT_BUDGET,
                 warm_fraction=DEFAULT_WARM_FRACTION,
                 prefetch_pool_workers=DEFAULT_PREFETCH_POOL_WORKERS):
        self.coarse_quantizer = faiss.read_index(ivf_centroids_path)
        self.temp_index = faiss.IndexFlatL2(self.coarse_quantizer.d)

        self.cluster_embedding_path = cluster_embedding_path
        self.result_path = result_path

        self.cache = ClusterCache(max_size=cache_size)
        self.cluster_generation_latency = {}
        self.cluster_file_size_mb = {}

        self.total_cluster_requests = 0
        self.total_cache_hits = 0
        self.idx_cnt = 1
        self.query_cnt = 1

        self.batch_id = 1
        self.warmupCnt = 1

        # Prefetch state
        self.prefetch_pool = ThreadPoolExecutor(max_workers=prefetch_pool_workers)
        self.prefetch_futures = {}  # {cid: Future}
        self.prefetch_wait_budget = prefetch_wait_budget
        self.warm_fraction = warm_fraction

        with open(self.result_path, mode='w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Seq", "Batch_Seq", "Duration", "EBT", "FLT", "CLT", "SLT", "NMC",
                "EVT", "PUT", "VST", "CHR", "QG", "NUM", "DRS", "PREFETCH"
            ])

    def write_results_to_csv(self, seq, bs, tt, ebt, flt, clt, slt, nmc, evt, put, vst, chr, qg, num, drs, prefetch):
        with open(self.result_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([seq, bs, tt, ebt, flt, clt, slt, nmc, evt, put, vst, chr, qg, num, drs, prefetch])

    def _submit_prefetch_if_needed(self, cid):
        cid = int(cid)
        if cid in self.cache.cache:
            return
        if cid in self.prefetch_futures:
            return
        # load_and_cache_cluster, cluster_id, self.cluster_embedding_path, self.cluster_generation_latency, self.cache)
        self.prefetch_futures[cid] = self.prefetch_pool.submit(
            load_and_cache_cluster,
            cid, self.cluster_embedding_path, self.cluster_generation_latency, self.cache
        )

    def prefetch_cluster(self, next_query, next_cluster_ids):
        self.cache.evict_if_needed_prefetch(len(next_cluster_ids))
        for cid in next_cluster_ids:
            try:
                self._submit_prefetch_if_needed(cid)
            except Exception as e:
                print(f"[Prefetch Error] Cluster {cid} failed: {e}")
        print(f"[Prefetch Submit] ({next_query}) -> {list(next_cluster_ids)}")

    def _profile_cluster_generation(self):
        for cluster_id in range(0, 100):
            try:
                start_time = time.time()
                cluster = f"cluster_{cluster_id}.npy"
                fp = self.cluster_embedding_path.joinpath(cluster)
                file_size = os.path.getsize(fp) / (1024.0 * 1024.0)
                np.load(fp)
                self.cluster_file_size_mb[cluster_id] = file_size
                np.load(fp)
                self.cluster_generation_latency[cluster_id] = time.time() - start_time
            except Exception:
                pass
        print("[Precompute] Cluster generation latencies computed successfully.")

    def search(self, query_text, model, sorted_duration, k, total_messages, dynamic_thread,
               prefetched_dict, num_thread, prefetch_granuality):
        # 1) encode
        encodeing_start_time_at_search = time.time()
        query_vector = model.encode([query_text])
        encodeing_end_time_at_search = time.time() - encodeing_start_time_at_search

        # 2) coarse lookup
        firstlookup_start_time_at_search = time.time()
        _, cluster_ids = self.coarse_quantizer.search(query_vector, k)
        firstlookup_end_time_at_search = time.time() - firstlookup_start_time_at_search
        cluster_ids = cluster_ids[0]

        self.query_cnt += 1
        self.total_cluster_requests += len(cluster_ids)

        # 3) 선택적 대기(타임박스)
        prefetch_waiting_start_time = time.time()
        waiting_targets = [int(cid) for cid in cluster_ids
                           if (int(cid) in self.prefetch_futures) and (self.cache.get(int(cid)) is None)]
        deadline = prefetch_waiting_start_time + self.prefetch_wait_budget
        for cid in waiting_targets:
            fut = self.prefetch_futures.get(cid)
            if fut is None:
                continue
            remain = deadline - time.time()
            if remain <= 0:
                break
            try:
                fut.result(timeout=max(0.0, remain))
            except concurrent.futures.TimeoutError:
                pass
            except Exception as e:
                print(f"[Prefetch Future Error] cid={cid}, {e}")
        prefetch_waiting_end_time = time.time() - prefetch_waiting_start_time

        # 4) 캐시 조회 / 미싱 수집
        cachelookup_start_time_at_search = time.time()
        temp_embeddings = []
        missing_clusters = []
        cache_hits = 0
        cache_miss = 0
        for cid in cluster_ids:
            emb = self.cache.get(int(cid))
            if emb is not None:
                temp_embeddings.append(emb)
                cache_hits += 1
            else:
                missing_clusters.append(int(cid))
                cache_miss += 1
        cachelookup_end_time_at_search = time.time() - cachelookup_start_time_at_search
        self.total_cache_hits += cache_hits

        # 5) 미싱 로딩
        secondlookup_start_time_at_search = time.time()
        lookup_file_size = 0
        evict_time = 0.0
        put_time = 0.0

        if missing_clusters:
            evict_start_time = time.time()
            self.cache.evict_if_needed(len(missing_clusters))
            evict_time = time.time() - evict_start_time

            num_workers = os.cpu_count() // 2 if dynamic_thread == "dynamic" else 1
            thread_assignments = optimized_clusters_placement(missing_clusters, self.cluster_file_size_mb, num_workers)

            put_start_time = time.time()
            with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
                futures = {
                    executor.submit(
                        load_and_cache_cluster,
                        cid, self.cluster_embedding_path,
                        self.cluster_generation_latency, self.cache
                    ): cid for cid in thread_assignments
                }
                for fut, cid in futures.items():
                    try:
                        result = fut.result()
                        if result is not None:
                            embeddings, file_size = result
                            temp_embeddings.append(embeddings)
                            lookup_file_size += file_size
                    except Exception as e:
                        print(f"[ERROR] Future for cluster {cid} failed: {e}")
            put_time = time.time() - put_start_time

        secondlookup_end_time_at_search = time.time() - secondlookup_start_time_at_search
        refined_lookup_file_size = int(lookup_file_size / (1024.0 * 1024.0))

        # 6) 벡터 검색
        vectorsearch_start_time_at_search = time.time()
        I, D = parallel_faiss_search(query_vector, temp_embeddings, k)
        vectorsearch_end_time_at_search = time.time() - vectorsearch_start_time_at_search
        total_duration = time.time() - prefetch_waiting_start_time

        # 7) 프리페치 트리거 (라스트-N)
        keys = list(sorted(prefetched_dict.keys()))
        boolean_prefetching = 0
        prefetching_start_time_at_search = time.time()

        for gid, meta in prefetched_dict.items():
            if query_text in meta["LAST_N"]:
                idx = keys.index(gid)
                if idx + 1 < len(keys):
                    nxt_gid = keys[idx + 1]
                    next_first_query = prefetched_dict[nxt_gid]["FQ"]
                    next_first_clusters = prefetched_dict[nxt_gid]["FQSET"]

                    boolean_prefetching = 1
                    # === granularity 분기 ===
                    if prefetch_granuality == "fg":
                        # FQ 전용 프리페치
                        self.prefetch_cluster(next_first_query, next_first_clusters)

                    elif prefetch_granuality == "balance":
                        # 다음 그룹 전체 쿼리의 클러스터 빈도를 집계, 상위(캐시 여유 슬롯)만 프리페치
                        q2c = prefetched_dict[nxt_gid].get("Q2C", {})
                        freq = defaultdict(int)
                        for _q, cset in q2c.items():
                            for cid in cset:
                                cid = int(cid)
                                if (self.cache.get(cid) is None) and (cid not in self.prefetch_futures):
                                    freq[cid] += 1
                        if freq:
                            free_slots = max(0, self.cache.max_size - len(self.cache.cache))
                            targets = [cid for cid, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)]
                            if free_slots > 0:
                                targets = targets[:free_slots]
                            if targets:
                                self.prefetch_cluster(next_first_query, targets)
                    # === granularity 분기 끝 ===
                break

        prefetching_async_end_time_at_search = time.time() - prefetching_start_time_at_search

        # 8) 로깅
        total_cache_num = cache_hits + cache_miss
        cache_hit_ratio = (cache_hits / total_cache_num) if total_cache_num > 0 else 0.0

        if self.warmupCnt > 20:
            self.write_results_to_csv(
                f"{self.idx_cnt}",
                f"{self.batch_id}",
                f"{total_duration:.4f}",
                f"{encodeing_end_time_at_search:.4f}",
                f"{firstlookup_end_time_at_search:.4f}",
                f"{cachelookup_end_time_at_search:.4f}",
                f"{secondlookup_end_time_at_search:.4f}",
                f"{len(missing_clusters)}",
                f"{evict_time:.4f}",
                f"{put_time:.4f}",
                f"{vectorsearch_end_time_at_search:.4f}",
                f"{cache_hit_ratio:.3%}",
                f"{sorted_duration:.4f}",
                f"{len(total_messages)}",
                f"{refined_lookup_file_size}",
                f"{prefetch_waiting_end_time:.4f}"
            )
            self.idx_cnt += 1
        self.warmupCnt += 1

        return I[0], D[0]

    def get_total_cache_hit_ratio(self):
        if self.total_cluster_requests == 0:
            return 0
        return self.total_cache_hits / self.total_cluster_requests


# ======================
# Utils
# ======================
def compute_jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batched_encode_and_cluster(batch, model, edgeRagSearcher, k):
    vectors = model.encode(batch, convert_to_numpy=True)
    results = []
    for query_text, vector in zip(batch, vectors):
        _, cluster_ids = edgeRagSearcher.coarse_quantizer.search(vector.reshape(1, -1), k)
        results.append((query_text, vector, frozenset(cluster_ids[0])))
    return results


def sort_queries_by_clustering(edgeRagSearcher, messages, model, float_value, k, linkage_value, jaccard_calculation):
    all_results = []
    batched_inputs = list(chunks(messages, max(1, len(messages) // 10 + 1)))
    num_workers = max(1, len(batched_inputs))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list(executor.map(lambda b: batched_encode_and_cluster(b, model, edgeRagSearcher, k), batched_inputs))
    for batch_result in futures:
        all_results.extend(batch_result)

    query_texts, query_vectors, cluster_sets = [], [], []
    cluster_id_mapping = {}

    for query_text, query_vector, cluster_ids_set in all_results:
        query_texts.append(query_text)
        query_vectors.append(query_vector)
        cluster_sets.append(cluster_ids_set)
        cluster_id_mapping[query_text] = cluster_ids_set

    num_queries = len(query_texts)
    jaccard_matrix = np.zeros((num_queries, num_queries))

    if jaccard_calculation == "vector":
        all_cluster_ids = set().union(*cluster_sets) if cluster_sets else set()
        cluster_id_list = sorted(list(all_cluster_ids))
        cluster_id_to_index = {cid: idx for idx, cid in enumerate(cluster_id_list)}
        num_clusters = len(cluster_id_list)

        binary_matrix = np.zeros((num_queries, num_clusters), dtype=np.uint8)
        for i, s in enumerate(cluster_sets):
            indices = [cluster_id_to_index[cid] for cid in s]
            binary_matrix[i, indices] = 1

        intersection = binary_matrix @ binary_matrix.T
        row_sums = binary_matrix.sum(axis=1)
        union = row_sums[:, None] + row_sums[None, :] - intersection
        jaccard_matrix = np.divide(intersection, union, where=union != 0)
        jaccard_distance_matrix = 1 - jaccard_matrix
    else:
        for i in range(num_queries):
            for j in range(num_queries):
                if i != j:
                    jaccard_matrix[i, j] = compute_jaccard_similarity(cluster_sets[i], cluster_sets[j])
        jaccard_distance_matrix = 1 - jaccard_matrix

    optimal_distance_threshold = 1.0 - float_value
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=optimal_distance_threshold,
        metric="precomputed", linkage=linkage_value
    )
    cluster_labels = clustering.fit_predict(jaccard_distance_matrix)

    # 그룹별 묶기
    cluster_groups = defaultdict(list)
    for query, assigned_cluster_id in zip(query_texts, cluster_labels):
        cluster_groups[assigned_cluster_id].append((query, cluster_id_mapping[query]))

    # prefetched_dict 구성 (Q2C 포함)
    prefetched_dict = {}
    keys_sorted = sorted(cluster_groups.keys())
    for key in keys_sorted:
        value = cluster_groups[key]  # [(query, set_of_ids), ...] (그룹 내부 순서는 그대로)
        queries = [q for (q, _) in value]
        last_n = set(queries[max(0, len(queries) - DEFAULT_LASTN_TO_TRIGGER):])
        # Q2C: 쿼리→클러스터 집합
        q2c = {q: set(cset) for (q, cset) in value}
        prefetched_dict[key] = {
            "FQ": value[0][0],
            "FQSET": value[0][1],
            "LQ": value[-1][0],
            "LQSET": value[-1][1],
            "LAST_N": last_n,
            "Q2C": q2c,
        }

    sorted_queries = sorted(zip(cluster_labels, query_texts, query_vectors), key=lambda x: x[0])
    return [(q, v) for _, q, v in sorted_queries], prefetched_dict


def kafka_search(topic_name, centroid_path, cluster_embedding_path, result_path,
                 float_value, nlist_s, cache_size_s, linkage_value,
                 dynamic_thread, jaccard_calculation, num_thread, prefetch_granuality):
    # 트래픽 설정
    avg_normal = 100
    time_interval = 1.0
    burst_duration = 5
    burst_chance = 0.1
    peak_scale = 3.0
    shape = 2.0
    burst_active = False
    burst_step = 0
    total_messages = []

    weibull_raw = np.random.weibull(shape, burst_duration)
    weibull_norm = weibull_raw / max(weibull_raw)
    burst_pattern = [int(avg_normal * (1 + (peak_scale - 1) * w)) for w in weibull_norm]

    # 모델/서처
    model = SentenceTransformer("all-MiniLM-L6-v2")
    edgeRagSearcher = EdgeRAGWithCache(
        centroid_path, cache_size_s, cluster_embedding_path, result_path,
        prefetch_wait_budget=DEFAULT_PREFETCH_WAIT_BUDGET,
        warm_fraction=DEFAULT_WARM_FRACTION,
        prefetch_pool_workers=DEFAULT_PREFETCH_POOL_WORKERS
    )
    edgeRagSearcher._profile_cluster_generation()

    # Kafka consumer
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers="163.239.199.205:9092",
        auto_offset_reset='earliest',
        max_poll_records=300000,
        max_partition_fetch_bytes=10485760,
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode('utf-8')
    )

    sliding_window = 1
    while True:
        if burst_active:
            target_count = burst_pattern[burst_step]
            burst_step += 1
            if burst_step >= burst_duration:
                burst_active = False
                burst_step = 0
                weibull_raw = np.random.weibull(shape, burst_duration)
                weibull_norm = weibull_raw / max(weibull_raw)
                burst_pattern = [int(avg_normal * (1 + (peak_scale - 1) * w)) for w in weibull_norm]
        else:
            if random.random() < burst_chance:
                burst_active = True
                target_count = burst_pattern[0]
                burst_step = 1
            else:
                target_count = random.randint(avg_normal // 2, avg_normal * 3 // 2)

        # 메시지 수신
        temp_messages = []
        while len(temp_messages) < target_count:
            remaining = target_count - len(temp_messages)
            msg_dict = consumer.poll(max_records=remaining, timeout_ms=1000)
            if not msg_dict:
                print("No more messages available, breaking out of the loop.")
            for _, messages in msg_dict.items():
                for msg in messages:
                    temp_messages.append(msg.value)
                    total_messages.append(msg.value)

            if sliding_window % 5 == 0:
                print("[Received message count per five seconds]:", len(total_messages))
                sorted_time = time.time()
                sorted_queries, prefetched_dict = sort_queries_by_clustering(
                    edgeRagSearcher, total_messages, model, float_value, nlist_s,
                    linkage_value, jaccard_calculation
                )
                sorted_duration = time.time() - sorted_time

                for query_text, query_vector in sorted_queries:
                    edgeRagSearcher.search(
                        query_text, model, sorted_duration, nlist_s,
                        total_messages, dynamic_thread, prefetched_dict,
                        num_thread, prefetch_granuality=prefetch_granuality
                    )

                total_messages.clear()
                sliding_window = 1
            else:
                sliding_window += 1
            time.sleep(time_interval)

        edgeRagSearcher.batch_id += 1


# ======================
# Search (병렬 클러스터 합치기)
# ======================
def parallel_faiss_search(query_vector, embeddings_list, k):
    def search_one_cluster(embeddings):
        index = faiss.IndexFlatL2(query_vector.shape[1])
        index.add(embeddings)
        return index.search(query_vector, k)

    results = []
    if len(embeddings_list) == 0:
        return [np.array([], dtype=np.int64)], [np.array([], dtype=np.float32)]

    with ThreadPoolExecutor(max_workers=len(embeddings_list)) as executor:
        futures = [executor.submit(search_one_cluster, emb) for emb in embeddings_list]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[ERROR] Parallel search error: {e}")

    if not results:
        print("Non results")
        return [np.array([], dtype=np.int64)], [np.array([], dtype=np.float32)]

    merged_D = np.hstack([r[0] for r in results])
    merged_I = np.hstack([r[1] for r in results])
    topk_idx = np.argsort(merged_D[0])[:k]
    return merged_I[0][topk_idx], merged_D[0][topk_idx]


# ======================
# Load & Cache
# ======================
def load_and_cache_cluster(cluster_id, cluster_embedding_path, cluster_generation_latency, cache):
    try:
        filename = cluster_embedding_path.joinpath(f"cluster_{cluster_id}.npy")
        lookup_file_size = os.path.getsize(filename)
        index = np.load(filename)
        latency = cluster_generation_latency.get(cluster_id, float("inf"))
        cache.put(cluster_id, index, latency)
        print(f"[Cluster Load] Cluster {cluster_id} loaded in full, Size={lookup_file_size/(1024*1024):.2f}MB")
        return index, lookup_file_size
    except Exception as e:
        print(f"[ERROR] Failed to load cluster {cluster_id}: {e}")
        return None


def drop_caches():
    try:
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=False)
    except Exception:
        pass


def load_and_cache_cluster_mmap(cluster_id, cluster_embedding_path, cluster_generation_latency, cache,
                                max_threads=10, min_shard_size=512, max_shard_size=2048,
                                warm_shards_first=True, warm_fraction=DEFAULT_WARM_FRACTION):
    try:
        filename = cluster_embedding_path.joinpath(f"cluster_{cluster_id}.npy")
        lookup_file_size = os.path.getsize(filename)
        mmap_array = np.load(filename, mmap_mode='r')
        total_vectors = mmap_array.shape[0]
        if total_vectors == 0:
            return None

        first_count = total_vectors if not warm_shards_first else max(1, int(total_vectors * warm_fraction))
        head = mmap_array[:first_count].copy()

        latency = cluster_generation_latency.get(cluster_id, float("inf"))
        cache.put(cluster_id, head, latency)

        if warm_shards_first and first_count < total_vectors:
            def fill_rest():
                try:
                    remain = total_vectors - first_count
                    def shard_copy(i):
                        s = first_count + i * min_shard_size
                        e = min(first_count + (i + 1) * min_shard_size, total_vectors)
                        return mmap_array[s:e].copy()
                    num_shards = math.ceil(remain / min_shard_size)
                    tails = []
                    with ThreadPoolExecutor(max_workers=min(max_threads, num_shards)) as ex:
                        futs = [ex.submit(shard_copy, i) for i in range(num_shards)]
                        for fu in as_completed(futs):
                            tails.append(fu.result())
                    if tails:
                        tail = np.vstack(tails)
                        merged = np.vstack([head, tail])
                        cache.put(cluster_id, merged, latency)
                except Exception as e_inner:
                    print(f"[Tail Fill Error] cid={cluster_id}, {e_inner}")

            threading.Thread(target=fill_rest, daemon=True).start()

        return head, lookup_file_size

    except Exception as e:
        print(f"[ERROR] Cluster {cluster_id} failed to load: {e}")
        return None


def optimized_clusters_placement(cluster_ids, cluster_size_dict, num_threads):
    sorted_ids = sorted(cluster_ids, key=lambda cid: cluster_size_dict.get(cid, 0.0), reverse=True)
    num_batches = max(1, math.ceil(len(sorted_ids) / max(1, num_threads)))
    batches = [[] for _ in range(num_batches)]
    batch_idx = 0
    for cid in sorted_ids:
        while len(batches[batch_idx]) >= max(1, num_threads):
            batch_idx += 1
            if batch_idx >= len(batches):
                batch_idx = len(batches) - 1
                break
        batches[batch_idx].append(cid)
    flat_execution_order = [cid for batch in batches for cid in batch]
    return flat_execution_order


# ======================
# Main
# ======================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 사용법: [arg0]Duration [arg1]python executor [arg2] python_script.py [arg3] dataset_name")
        sys.exit(1)

    print("경로 확인")
    dataset_name = sys.argv[1]
    float_value = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    cluster_size = int(sys.argv[3])
    nlist_s = int(sys.argv[4])
    cache_size_s = int(sys.argv[5])
    linkage_value = sys.argv[6]
    dynamic_thread = sys.argv[7]         # "dynamic" or else
    jaccard_calculation = sys.argv[8]    # "vector" or else
    num_thread = int(sys.argv[9])
    prefetch_granuality = sys.argv[10] if len(sys.argv) > 10 else "fg"  # "fg" or "balance"

    if dataset_name == "hotpotqa":
        topic_name = dataset_name
    else:
        topic_name = dataset_name + "_query"

    cluster_embedding_path = pathlib.Path(__file__).parent.absolute().joinpath("disk_clusters", f"{dataset_name}_{cluster_size}")
    print(f"✅ 클러스터 npy 파일 경로: {cluster_embedding_path}")

    result_dir = pathlib.Path(__file__).parent.absolute().joinpath(
        "europar_results", dataset_name, "prefetch",
        f"{cluster_size}", f"{float_value}", f"{linkage_value}", f"{dynamic_thread}", f"{jaccard_calculation}", f"{prefetch_granuality}"
    )
    result_filename = f"{timestamp}_nlist_{nlist_s}_cache_{cache_size_s}_thread_{num_thread}.csv"

    os.makedirs(result_dir, exist_ok=True)
    result_path = result_dir.joinpath(result_filename)
    print(f"✅ 결과 csv 파일 경로: {result_path}")

    index_dir = pathlib.Path(__file__).parent.absolute().joinpath("index", dataset_name)
    inf_centroids_path = index_dir.joinpath(f"{dataset_name}_centroids.index")
    faiss_inf_centroids_path = str(inf_centroids_path)
    print(f"✅ first level index 파일 경로: {faiss_inf_centroids_path}")

    try:
        kafka_search(
            topic_name, faiss_inf_centroids_path, cluster_embedding_path, result_path,
            float_value, nlist_s, cache_size_s, linkage_value,
            dynamic_thread, jaccard_calculation, num_thread, prefetch_granuality=prefetch_granuality
        )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f'file name: {str(fname)}')
        print(f'error type: {str(exc_type)}')
        print(f'error msg: {str(e)}')
        print(f'line number: {str(exc_tb.tb_lineno)}')
        sys.exit(1)
