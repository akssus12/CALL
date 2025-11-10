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
from collections import OrderedDict
import csv, psutil, json, math
import pathlib, random, torch, threading
import subprocess ############ 0620 nomerge index -> parallel indexing & search ############

# Global variables
minibatch_model = None
buffered_queries = []
buffered_vectors = []
n_clusters = 10
model_name = "all-MiniLM-L6-v2"
timestamp = time.strftime("%m%d_%H%M%S")

class ClusterCache:
    def __init__(self, max_size):        
        self.cache = OrderedDict()  # Store embeddings {cluster_id: {"embeddings": np.array, "latency": float, "count": int}}
        self.max_size = max_size

    def _calculate_eviction_priority(self, cluster_id):
        return self.cache[cluster_id]["latency"] * self.cache[cluster_id]["count"]
    
    ############ 0628 prefetch_if_needed_prefetch ############
    def evict_if_needed_prefetch(self, new_cluster_count):
        current_size = len(self.cache)
        if current_size + new_cluster_count > self.max_size:
            num_to_evict = (current_size + new_cluster_count) - self.max_size
            # print(f"[Cache Eviction] Removing {num_to_evict} entries.")

            eviction_candidates = sorted(self.cache.keys(), key=self._calculate_eviction_priority)[:num_to_evict]

            for cluster_id in eviction_candidates:
                del self.cache[cluster_id]
                # print(f"[Cache Eviction] Evicted Cluster {cluster_id}")
            ############ 0620 nomerge index -> parallel indexing & search ############
            threading.Thread(target=gc.collect, daemon=True).start()  # [변경된 부분] 백그라운드에서 GC 실행
            #drop_caches()
            ############ 0620 nomerge index -> parallel indexing & search ############
    ############ 0628 prefetch_if_needed_prefetch ############

    def evict_if_needed(self, new_cluster_count):
        current_size = len(self.cache)
        if current_size + new_cluster_count > self.max_size:
            num_to_evict = (current_size + new_cluster_count) - self.max_size
            # print(f"[Cache Eviction] Removing {num_to_evict} entries.")

            eviction_candidates = sorted(self.cache.keys(), key=self._calculate_eviction_priority)[:num_to_evict]

            for cluster_id in eviction_candidates:
                del self.cache[cluster_id]
                # print(f"[Cache Eviction] Evicted Cluster {cluster_id}")
            ############ 0620 nomerge index -> parallel indexing & search ############
            gc.collect()
            drop_caches()
            ############ 0620 nomerge index -> parallel indexing & search ############
            
    def get(self, cluster_id):        
        if cluster_id in self.cache:
            self.cache[cluster_id]["count"] += 1  # Increase visit count
            self.cache.move_to_end(cluster_id)  # Mark as recently used
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
                # print(f"[Cache Eviction] Removing Cluster {eviction_target}")
                del self.cache[eviction_target]

            self.cache[cluster_id] = {"embeddings": embeddings, "latency": generation_latency, "count": 1}
            self.cache.move_to_end(cluster_id)  # Mark as recently used

class EdgeRAGWithCache:
    def __init__(self, ivf_centroids_path, cache_size, cluster_embedding_path, result_path):
        self.coarse_quantizer = faiss.read_index(ivf_centroids_path)
        self.temp_index = faiss.IndexFlatL2(self.coarse_quantizer.d)

        self.cluster_embedding_path = cluster_embedding_path

        self.result_path = result_path

        self.cache = ClusterCache(max_size=cache_size)
        self.cluster_generation_latency = {}  # Store precomputed latencies
        self.cluster_file_size_mb = {} ############ 0627 assign_clusters_threads ############

        self.total_cluster_requests = 0
        self.total_cache_hits = 0
        self.idx_cnt = 1
        self.query_cnt = 1 # for motivation1

        self.batch_id = 1
        self.warmupCnt = 1 # First msgs are used for warmup(1: warum-up, 2: experiment start)

        self.query_buffer = []  # Temporary query storage for sorting

        # [변경된 부분]
        self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8) ############ 0628 prefetch_if_needed_prefetch ############
        self.prefetch_future = None  # Track prefetch task

        # [변경된 부분] open(self.result_path, mode='w', newline="") ~ write_results_to_csv
        # [Parameters]
        # Seq: Sequence number of the query
        # Batch_Seq: Batch sequence number for grouping
        # Duration: Total duration of the search operation
        # EBT: Encoding time for the query
        # FLT: First lookup time in the coarse quantizer
        # CLT: Cache lookup time for precomputed clusters
        # SLT: Second lookup time for missing clusters
        # NMC: Number of missing clusters that needed to be generated
        # EVT: Eviction time for the cache
        # PUT: Time taken to put new clusters into the cache
        # MIT: Time taken to merge the embeddings into the FAISS index
        # VST: Vector search time in the FAISS index
        # CHR: Cache hit ratio for the query
        # QG:  Grouping time (if applicable)
        # NUM: Number of messages processed in this batch
        # DRS: Disk read size for the lookup files (in MB)
        # PREFETCH: Prefetching time for the next cluster group

        ############ 0620 nomerge index -> parallel indexing & search ############
        with open(self.result_path, mode='w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Seq", "Batch_Seq", "Duration", "EBT", "FLT", "CLT", "SLT", "NMC", "EVT", "PUT", "VST", "CHR", "QG", "NUM", "DRS", "PREFETCH", "LBT"])
    
    def write_results_to_csv(self, seq, bs, tt, ebt, flt, clt, slt, nmc, evt, put, vst, chr, qg, num, drs, prefetch, lbt):
        with open(self.result_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([seq, bs, tt, ebt, flt, clt, slt, nmc, evt, put, vst, chr, qg, num, drs, prefetch, lbt])
        ############ 0620 nomerge index -> parallel indexing & search ############
    
    # [변경된 부분]
    def prefetch_cluster(self, next_query, next_cluster_ids):
        ############ 0628 prefetch_if_needed_prefetch ############
        self.cache.evict_if_needed_prefetch(len(next_cluster_ids))

        def prefetch_single_cluster(cluster_id):
            if cluster_id not in self.cache.cache:
                try:
                    converted_cluster_id = int(str(cluster_id))
                    load_and_cache_cluster_mmap(
                        converted_cluster_id,
                        self.cluster_embedding_path,
                        self.cluster_generation_latency,
                        self.cache
                    )
                except Exception as e:
                    print(f"[Prefetch Error] Cluster {cluster_id} failed: {e}")

        with ThreadPoolExecutor(max_workers=len(next_cluster_ids)) as executor:
            futures = [executor.submit(prefetch_single_cluster, cid) for cid in next_cluster_ids]
            for future in as_completed(futures):
                _ = future.result()
        ############ 0628 prefetch_if_needed_prefetch ############

        # for cluster_id in next_cluster_ids:
        #     if cluster_id not in self.cache.cache:
        #         converted_cluster_id = int(str(cluster_id))
        #         cluster_filename = f"cluster_{cluster_id}.npy"
        #         cluster_embedding_path = self.cluster_embedding_path.joinpath(cluster_filename)
        #         new_embeddings = np.load(cluster_embedding_path)

        #         generation_latency = self.cluster_generation_latency.get(converted_cluster_id, float("inf"))
        #         self.cache.put(cluster_id, new_embeddings, generation_latency)

        print(f"[Prefetching Query: {next_query}] Completed for clusters:", next_cluster_ids)
    # [변경된 부분]

    def _profile_cluster_generation(self):
        for cluster_id in range(0,100):
            start_time = time.time()
            cluster = f"cluster_{cluster_id}.npy"
            cluster_embedding = self.cluster_embedding_path.joinpath(cluster)
            ############ 0627 assign_clusters_threads ############
            file_size = os.path.getsize(cluster_embedding) / (1024.0 * 1024.0)
            np.load(cluster_embedding)
            self.cluster_file_size_mb[cluster_id] = file_size
            ############ 0627 assign_clusters_threads ############
            np.load(cluster_embedding)
            self.cluster_generation_latency[cluster_id] = time.time() - start_time
        print("[Precompute] Cluster generation latencies computed successfully.")       

    def search(self, query_text, model, sorted_duration, k, total_messages, dynamic_thread, prefetched_dict, num_thread):
        ############ 0627 assign_clusters_threads ############
        encodeing_start_time_at_search = time.time()
        query_vector = model.encode([query_text])
        encodeing_end_time_at_search = time.time() - encodeing_start_time_at_search
        ############ 0627 assign_clusters_threads ############        

        firstlookup_start_time_at_search = time.time()
        _, cluster_ids = self.coarse_quantizer.search(query_vector, k)
        firstlookup_end_time_at_search = time.time() - firstlookup_start_time_at_search
        cluster_ids = cluster_ids[0]  # Extract cluster IDs, return numpy.int64, so it has to convert to int

        # print(f"Seq, {self.query_cnt}, {cluster_ids}")
        self.query_cnt += 1

        self.total_cluster_requests += len(cluster_ids)
        cache_hits = 0  # Track hits per query
        cache_miss = 0
        
        temp_embeddings = [] # embeddings for selective search
        missing_clusters = [] # need for online generation

        ############ 0627 prefetch_if_needed_prefetch ############
        prefetch_waiting_start_time = time.time()
        if self.prefetch_future is not None:
            self.prefetch_future.result()
        prefetch_waiting_end_time = time.time() - prefetch_waiting_start_time
        ############ 0627 prefetch_if_needed_prefetch ############

        cachelookup_start_time_at_search = time.time()
        for cluster_id in cluster_ids:
            cached_embeddings = self.cache.get(cluster_id)
            
            if cached_embeddings is not None:
                #print(f"[Cache Hit] Using Precomputed Cluster {cluster_id}")
                temp_embeddings.append(cached_embeddings)
                cache_hits += 1
            else:
                #print(f"[Cache Miss] Cluster {cluster_id} needs embedding generation.")
                missing_clusters.append(cluster_id)
                cache_miss += 1
        cachelookup_end_time_at_search = time.time() - cachelookup_start_time_at_search
        self.total_cache_hits += cache_hits

        secondlookup_start_time_at_search = time.time()
        lookup_file_size = 0
        evict_time = 0 # [변경된 부분]
        put_time = 0 # [변경된 부분]
        lb_time = 0  
        if missing_clusters is not None:
            evict_start_time = time.time()
            self.cache.evict_if_needed(len(missing_clusters)) # Evict multiple entries if needed before inserting new ones
            evict_time = time.time() - evict_start_time

            ############ 0627 assign_clusters_threads ############
            num_workers = os.cpu_count() / 2
            if dynamic_thread == "dynamic" and len(missing_clusters) > 0:  # ours
                num_workers = 8 # len(missing_clusters)
                lb_start_time = time.time()
                thread_assignments = optimized_clusters_placement(missing_clusters, self.cluster_file_size_mb, num_workers)
                lb_time = time.time() - lb_start_time
            else:
                num_workers = 1
                thread_assignments = missing_clusters
            print(f"[Parallel] Starting to load {len(missing_clusters)} missing clusters with {num_workers} workers...")
            ############ 0627 assign_clusters_threads ############
            
            put_start_time = time.time()

            if len(missing_clusters) > 0:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(load_and_cache_cluster, cluster_id, self.cluster_embedding_path, self.cluster_generation_latency, self.cache): cluster_id for cluster_id in thread_assignments}
                    #futures = {executor.submit(load_and_cache_cluster_mmap, cluster_id, self.cluster_embedding_path, self.cluster_generation_latency, self.cache, max_threads=1, min_shard_size=512, max_shard_size=2048): cluster_id for cluster_id in thread_assignments}

                for future in as_completed(futures):
                    cluster_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            ############ 0620 nomerge index -> parallel indexing & search ############
                            embeddings, file_size = result
                            temp_embeddings.append(embeddings)
                            lookup_file_size += file_size
                            ############ 0620 nomerge index -> parallel indexing & search ############
                    except Exception as e:
                        print(f"[ERROR] Future for cluster {cluster_id} failed: {e}")
            else:
                pass
            
            put_time = time.time() - put_start_time
            # [변경된 부분]
            
        secondlookup_end_time_at_search = time.time() - secondlookup_start_time_at_search
        refined_lookup_file_size = int(lookup_file_size / (1024.0 * 1024.0))        

        vectorsearch_start_time_at_search = time.time()
        ############ 0620 nomerge index -> parallel indexing & search ############
        I, D = parallel_faiss_search(query_vector, temp_embeddings, k)
        ############ 0620 nomerge index -> parallel indexing & search ############
        vectorsearch_end_time_at_search = time.time() - vectorsearch_start_time_at_search
        total_duration = time.time() - firstlookup_start_time_at_search ############ 0627 assign_clusters_threads ############
        
        total_cache_num = cache_hits + cache_miss
        cache_hit_ratio = cache_hits / total_cache_num

        keys = list(sorted(prefetched_dict.keys()))
        
        nxt_key = 0
        boolean_prefetching = 0
        prefetching_start_time_at_search = time.time()
        for key in prefetched_dict:
            current_index = keys.index(key)
            if current_index + 1 < len(keys): # avoid -- list index out of range
                nxt_key = keys[current_index+1]
            # print(f"[PP] Current Query: {query_text}, Last Query: {prefetched_dict[key]['LQ']}, Next Cluster: {prefetched_dict[nxt_key]['FQ']}")
            if query_text == prefetched_dict[key]['LQ']: # last query of current cluster group -> This is timing to prefetch because cluster group will be switched
                # print(f"[P__P] Current Query: {query_text}, Last Query: {prefetched_dict[key]['LQ']}, Next Cluster: {prefetched_dict[nxt_key]['FQ']}")
                boolean_prefetching = 1
                first_query_next_cluster = prefetched_dict[nxt_key]['FQ'] # first query of next cluster group
                first_query_cluster_ids_next_cluster = prefetched_dict[nxt_key]['FQSET'] # first query's clusterIDs of next cluster group
                self.prefetch_future = self.prefetch_executor.submit(self.prefetch_cluster, first_query_next_cluster, first_query_cluster_ids_next_cluster)
                break
        prefetching_async_end_time_at_search = time.time() - prefetching_start_time_at_search

        ############ 0620 nomerge index -> parallel indexing & search ############
        if self.warmupCnt > 20:
            self.write_results_to_csv(f"{self.idx_cnt}",
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
                                      f"{prefetch_waiting_end_time:.4f}",
                                      f"{lb_time:.4f}"
                                      )
        ############ 0620 nomerge index -> parallel indexing & search ############
            self.idx_cnt += 1
        self.warmupCnt += 1

        # threading.Thread(target=gc.collect, daemon=True).start()  # [변경된 부분] 백그라운드에서 GC 실행

        return I[0], D[0]
    
    def get_total_cache_hit_ratio(self):
        if self.total_cluster_requests == 0:
            return 0
        return self.total_cache_hits / self.total_cluster_requests # overall cache utilization

def compute_jaccard_similarity(set1, set2):
        if not set1 or not set2:
            return 0  # No overlap
        return len(set1 & set2) / len(set1 | set2)

# [변경된 부분]
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
# [변경된 부분]

# [변경된 부분]
def batched_encode_and_cluster(batch, model, edgeRagSearcher, k):
    start_batched_encode_cluster_time = time.time()
    vectors = model.encode(batch, convert_to_numpy=True)
    results = []
    for query_text, vector in zip(batch, vectors):
        _, cluster_ids = edgeRagSearcher.coarse_quantizer.search(vector.reshape(1, -1), k)
        results.append((query_text, vector, frozenset(cluster_ids[0])))
    end_batched_encode_cluster_time = time.time() - start_batched_encode_cluster_time
    #print(f"[Batched Encoding and Clustering] Processed {len(batch)} queries in {end_batched_encode_cluster_time:.3f} seconds.")
    return results
# [변경된 부분]

def sort_queries_by_clustering(edgeRagSearcher, messages, model, float_value, k, linkage_value, jaccard_calculation):
    all_results = []

    # [변경된 부분]    
    batched_inputs = list(chunks(messages, len(messages) // 10 + 1))  # Split messages into batches and distribute them across threads
    num_workers = len(batched_inputs)

    start_batched_encode_cluster_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list(executor.map(lambda b: batched_encode_and_cluster(b, model, edgeRagSearcher, k), batched_inputs))

    for batch_result in futures:
        all_results.extend(batch_result)
    end_batched_encode_cluster_time = time.time() - start_batched_encode_cluster_time
    # [변경된 부분]

    query_texts = []
    query_vectors = []
    cluster_sets = []
    cluster_id_mapping = {}

    for query_text, query_vector, cluster_ids_set in all_results:
        query_texts.append(query_text)
        query_vectors.append(query_vector)
        cluster_sets.append(cluster_ids_set)
        cluster_id_mapping[query_text] = cluster_ids_set    
    
    num_queries = len(query_texts)
    jaccard_matrix = np.zeros((num_queries, num_queries))
    
    # compute_jaccard_similarity 계산 오버헤드가 쿼리 개수에 비례하여 증가함. 여기를 병렬화할 수 있는가?
    # [변경된 부분]
    start_compute_jaccard_similarity_time = time.time()
    
    if jaccard_calculation == "vector": # 최적화된 방식(scalable degisn)
        # 1. 집합 전체 클러스터 id 찾기
        all_cluster_ids = set().union(*cluster_sets)
        cluster_id_list = sorted(list(all_cluster_ids))  # 안정성 위해 정렬

        cluster_id_to_index = {cid: idx for idx, cid in enumerate(cluster_id_list)}
        num_clusters = len(cluster_id_list)

        # 2. 바이너리 행렬 생성
        binary_matrix = np.zeros((num_queries, num_clusters), dtype=np.uint8)
        for i, s in enumerate(cluster_sets):
            indices = [cluster_id_to_index[cid] for cid in s]
            binary_matrix[i, indices] = 1

        # 3. 교집합 계산 (벡터 내적)
        intersection = binary_matrix @ binary_matrix.T

        # 4. 합집합 계산
        row_sums = binary_matrix.sum(axis=1)
        union = row_sums[:, None] + row_sums[None, :] - intersection

        # 5. 자카드 계산
        jaccard_matrix = np.divide(intersection, union, where=union != 0)

        # 6. 거리 행렬 생성
        jaccard_distance_matrix = 1 - jaccard_matrix
    else: # 기존 방식
        #print(f"compute_jaccard_similarity start!")
        for i in range(num_queries):
            for j in range(num_queries):
                if i != j:
                    jaccard_matrix[i, j] = compute_jaccard_similarity(cluster_sets[i], cluster_sets[j])
        #print(f"compute_jaccard_similarity end!")
    
    end_compute_jaccard_similarity_time = time.time() - start_compute_jaccard_similarity_time
    # [변경된 부분]

    jaccard_distance_matrix = 1 - jaccard_matrix
    
    optimal_distance_threshold = 1.0 - float_value # default : 0.3 -> jaccard distance = 70% 다르다는 이야기.
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=optimal_distance_threshold, metric="precomputed", linkage=linkage_value)
    cluster_labels = clustering.fit_predict(jaccard_distance_matrix)
        
    cluster_groups = {}
    
    for query, assigned_cluster_id in zip(query_texts, cluster_labels):
        if assigned_cluster_id not in cluster_groups:
            cluster_groups[assigned_cluster_id] = []

        cluster_groups[assigned_cluster_id].append((query, cluster_id_mapping[query]))  # Store FAISS cluster IDs

    prefetched_dict = {
        key: {
            "FQ": value[0][0],
            "FQSET": value[0][1],
            "LQ": value[-1][0],
            "LQSET": value[-1][1],
        }
        for key, value in cluster_groups.items()
    }
    
    sorted_queries = sorted(zip(cluster_labels, query_texts, query_vectors), key=lambda x: x[0]) # negliable overhead

    return [(q, v) for _, q, v in sorted_queries], prefetched_dict

def kafka_search(topic_name, centroid_path, cluster_embedding_path, result_path, float_value, nlist_s, cache_size_s, linkage_value, dynamic_thread, jaccard_calculation, num_thread):
    # [변경된 부분]
    ################## For traffic settings(Weibull distribution) ##################
    avg_normal=100
    time_interval=1.0
    burst_duration=5
    burst_chance=0.1 # burst chance가 높으면 더 많은 burst 상황 발생하는 것을 의미함.
    peak_scale = 3.0
    shape=2.0
    burst_active = False
    burst_step = 0

    total_messages = []

    # Create a normalized Weibull pattern (0~1)
    weibull_raw = np.random.weibull(shape, burst_duration)
    weibull_norm = weibull_raw / max(weibull_raw)
    burst_pattern = [int(avg_normal * (1 + (peak_scale - 1) * w)) for w in weibull_norm]
    ##################################################################################
    # [변경된 부분]   
    
    ################## For model settings ############################################
    model = SentenceTransformer("all-MiniLM-L6-v2")
    edgeRagSearcher = EdgeRAGWithCache(centroid_path, cache_size_s, cluster_embedding_path, result_path)
    edgeRagSearcher._profile_cluster_generation() # Pre-compute cluster generation() before search phase
    ##################################################################################

    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers="163.239.199.205:9092",
        auto_offset_reset='earliest',  # 가장 처음부터 읽기 ('latest'로 설정하면 최신 메시지만 읽음)
        max_poll_records=300000,
        max_partition_fetch_bytes=10485760,
        enable_auto_commit=True,  # 자동 커밋 활성화
        value_deserializer=lambda x: x.decode('utf-8')  # 바이트 데이터를 문자열로 변환
    )

    # [변경된 부분]
    sliding_window = 1

    while True:
        if burst_active:
            target_count = burst_pattern[burst_step]
            #print(f"[BURST] Weibull Step {burst_step+1}/{burst_duration} → {target_count} msgs")
            burst_step += 1
            if burst_step >= burst_duration:
                burst_active = False
                burst_step = 0                
                weibull_raw = np.random.weibull(shape, burst_duration)
                weibull_norm = weibull_raw / max(weibull_raw)
                burst_pattern = [int(avg_normal * (1 + (peak_scale - 1) * w)) for w in weibull_norm]
        else:
            if random.random() < burst_chance:
                #print("[BURST START] Weibull burst initiated!")
                burst_active = True
                target_count = burst_pattern[0]
                burst_step = 1
            else:
                target_count = random.randint(avg_normal // 2, avg_normal * 3 // 2)

        temp_messages = [] # 메시지 수신용 임시 변수
        
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
                sorted_queries, prefetched_dict = sort_queries_by_clustering(edgeRagSearcher, total_messages, model, float_value, nlist_s, linkage_value, jaccard_calculation)
                sorted_duration = time.time() - sorted_time
                
                for query_text, query_vector in sorted_queries:
                    edgeRagSearcher.search(query_text, model, sorted_duration, nlist_s, total_messages, dynamic_thread, prefetched_dict, num_thread)
                
                total_messages.clear()
                sliding_window = 1
            else:
                sliding_window += 1
            time.sleep(time_interval)
        # [변경된 부분]

        edgeRagSearcher.batch_id += 1

############ 0620 nomerge index -> parallel indexing & search ############
def parallel_faiss_search(query_vector, embeddings_list, k):
    def search_one_cluster(embeddings):        
        index = faiss.IndexFlatL2(query_vector.shape[1])
        index.add(embeddings)
        return index.search(query_vector, k)

    results = []
    with ThreadPoolExecutor(max_workers=len(embeddings_list)) as executor:
        futures = [executor.submit(search_one_cluster, emb) for emb in embeddings_list]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[ERROR] Parallel search error: {e}")

    if not results:
        print("Non results")
        return [], []
    
    merged_D = np.hstack([r[0] for r in results])    
    merged_I = np.hstack([r[1] for r in results])    
    topk_idx = np.argsort(merged_D[0])[:k]    
    
    return merged_I[0][topk_idx], merged_D[0][topk_idx]
############ 0620 nomerge index -> parallel indexing & search ############

############ 0620 nomerge index -> parallel indexing & search ############
def load_and_cache_cluster(cluster_id, cluster_embedding_path, cluster_generation_latency, cache):
    try:
        filename = cluster_embedding_path.joinpath(f"cluster_{cluster_id}.npy")
        lookup_file_size = 0
        lookup_file_size += os.path.getsize(cluster_embedding_path.joinpath(filename))
        index = np.load(filename)
        latency = cluster_generation_latency.get(cluster_id, float("inf"))
        cache.put(cluster_id, index, latency)
        return index, lookup_file_size
    except Exception as e:
        print(f"[ERROR] Failed to load cluster {cluster_id}: {e}")
        return None
############ 0620 nomerge index -> parallel indexing & search ############

############ 0624 determine_optimal_threads ############
def determine_optimal_threads(missing_clusters, cluster_latency_dict, latency_bound, avg_cluster_size_mb=4.0, alpha=0.5):    
    L = sum(cluster_latency_dict.get(cid, float("inf")) for cid in missing_clusters) # Total estimated loading latency
    M = len(missing_clusters) # Number of missing clusters
    avg_latency = L / M if M > 0 else 0.1
    # print(f"[Adaptive Threading] Total Latency: {L:.3f}s, Avg Latency: {avg_latency:.3f}s, Missing Clusters: {M}")

    # Resource estimation
    cpu_count = psutil.cpu_count(logical=False)  # physical cores preferred
    mem_available = psutil.virtual_memory().available / (1024**2)  # MB
    # print(f"[Adaptive Threading] CPU Count: {cpu_count}, Available Memory: {mem_available:.1f}MB")

    n_latency = int(L / (latency_bound * avg_latency)) + 1 # 0.404 / 0.040 = 
    n_latency = min(M, n_latency)

    n_mem = int(mem_available / (alpha * avg_cluster_size_mb))
    # print(f"[Adaptive Threading] Estimated Threads (Latency): {n_latency}, Estimated Threads (Memory): {n_mem}")

    # Final thread count is min of all bounds
    n_final = max(1, min(M, cpu_count, n_latency, n_mem))

    # print(f"[Adaptive Threading] Total Latency: {L:.3f}s, CPU: {cpu_count}, Mem: {mem_available:.1f}MB, Avg_Latency: {avg_latency:.3f}s")
    # print(f"[Adaptive Threading] Threads (Latency): {n_latency}, Threads (Mem): {n_mem} → Final Threads: {n_final}")
    
    return n_final
############ 0624 determine_optimal_threads ############

def drop_caches():
    subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
############ 0620 nomerge index -> parallel indexing & search ############

############ 0618 load_and_cache_cluster ############
def load_and_cache_cluster(cluster_id, cluster_embedding_path, cluster_generation_latency, cache):
    try:
        load_start_time = time.time()
        filename = cluster_embedding_path.joinpath(f"cluster_{cluster_id}.npy")
        lookup_file_size = 0
        lookup_file_size += os.path.getsize(cluster_embedding_path.joinpath(filename))
        index = np.load(filename)
        load_end_time = time.time() - load_start_time
        cgl_get_start_time = time.time()
        latency = cluster_generation_latency.get(cluster_id, float("inf"))
        cgl_get_end_time = time.time() - cgl_get_start_time
        cache_put_start_time = time.time()
        cache.put(cluster_id, index, latency)
        cache_end_time = time.time() - cache_put_start_time
        print(f"[Cluster Load] Cluster {cluster_id} loaded in {load_end_time:.4f}s, File Size: {lookup_file_size / (1024.0 * 1024.0):.2f} MB")        
        return index, lookup_file_size
    except Exception as e:
        print(f"[ERROR] Failed to load cluster {cluster_id}: {e}")
        return None
############ 0618 load_and_cache_cluster ############

############ 0626 mmap read ############
def load_and_cache_cluster_mmap(cluster_id, cluster_embedding_path, cluster_generation_latency, cache, max_threads=10, min_shard_size=512, max_shard_size=2048):
    """
    Optimized cluster loader using mmap and parallel decoding.
    This function fully replaces the original load_and_cache_cluster().
    """
    try:
        filename = cluster_embedding_path.joinpath(f"cluster_{cluster_id}.npy")
        lookup_file_size = os.path.getsize(filename)

        # Step 1: Memory-map the file (lazy loading)
        mmap_array = np.load(filename, mmap_mode='r')        
        total_vectors = mmap_array.shape[0]        

        # Dynamically determine shard size and number of threads
        for shard_size in range(min_shard_size, max_shard_size + 1, 100):
            est_threads = math.ceil(total_vectors / shard_size)
            if est_threads <= max_threads:
                actual_shard_size = shard_size
                thread_count = est_threads
                break
        else:
            thread_count = max_threads
            actual_shard_size = math.ceil(total_vectors / thread_count)
        # print(f"[Cluster Load] Cluster {cluster_id} - Total Vectors: {total_vectors}, Shard Size: {actual_shard_size}, Threads: {thread_count}")

        # Step 3: Parallel decode shards
        embeddings = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                #executor.submit(lambda i=i: mmap_array[i*shard_size : min((i+1)*shard_size, total_vectors)].copy() #, shard_idx)
                executor.submit(lambda i=i: mmap_array[i*actual_shard_size:min((i+1)*actual_shard_size, total_vectors)].copy())
                #executor.submit(log_and_copy, i, actual_shard_size, total_vectors, mmap_array)
                for i in range(thread_count)
            ]
            for future in as_completed(futures):
                shard = future.result()
                embeddings.append(shard)

        # Step 4: Combine all decoded vectors
        merged_embeddings = np.vstack(embeddings)

        # Step 5: Retrieve latency and put to cache
        latency = cluster_generation_latency.get(cluster_id, float("inf"))
        cache.put(cluster_id, merged_embeddings, latency)

        # print(f"[Cluster Load] Cluster {cluster_id} loaded with mmap + parallel decode. File Size: {lookup_file_size / (1024.0 * 1024.0):.2f} MB, Vectors: {total_vectors}")
        return merged_embeddings, lookup_file_size

    except Exception as e:
        print(f"[ERROR] Cluster {cluster_id} failed to load: {e}")
        return None
############ 0626 mmap read ############

############ 0627 assign_clusters_threads ############
def optimized_clusters_placement(cluster_ids, cluster_size_dict, num_threads):    
    sorted_ids = sorted(cluster_ids, key=lambda cid: cluster_size_dict.get(cid, 0.0), reverse=True)
    num_batches = math.ceil(len(sorted_ids) / num_threads)

    batches = [[] for _ in range(num_batches)]
    batch_idx = 0
    for cid in sorted_ids:
        while len(batches[batch_idx]) >= num_threads:
            batch_idx += 1
        batches[batch_idx].append(cid)
    
    flat_execution_order = [cid for batch in batches for cid in batch]
    return flat_execution_order
############ 0627 assign_clusters_threads ############

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
    dynamic_thread = sys.argv[7] # [변경된 부분]
    jaccard_calculation = sys.argv[8] # [변경된 부분]
    #topic_name = dataset_name + "_query" # [변경된 부분]
    
    if dataset_name == "hotpotqa":
        topic_name = dataset_name
    else:
        topic_name = dataset_name + "_query"
    num_thread = int(sys.argv[9]) # [변경된 부분]
    
    cluster_embedding_path = pathlib.Path(__file__).parent.absolute().joinpath("disk_clusters", f"{dataset_name}_{cluster_size}")
    print(f"✅ 클러스터 npy 파일 경로: {cluster_embedding_path}")

    # [변경된 부분]
    result_dir = pathlib.Path(__file__).parent.absolute().joinpath("europar_results", dataset_name, "prefetch", f"{cluster_size}", f"{float_value}", f"{linkage_value}", f"{dynamic_thread}", f"{jaccard_calculation}")
    result_filename = f"{timestamp}_nlist_{nlist_s}_cache_{cache_size_s}_thread_{num_thread}.csv"
    # [변경된 부분]

    os.makedirs(result_dir, exist_ok=True)
    result_path = result_dir.joinpath(result_filename)
    print(f"✅ 결과 csv 파일 경로: {result_path}")

    index_dir = pathlib.Path(__file__).parent.absolute().joinpath("index", dataset_name)
    inf_centroids_path = index_dir.joinpath(f"{dataset_name}_centroids.index") # 수정 0606
    faiss_inf_centroids_path = str(inf_centroids_path)
    print(f"✅ first level index 파일 경로: {faiss_inf_centroids_path}")

    try:
        # [변경된 부분]
        kafka_search(topic_name, faiss_inf_centroids_path, cluster_embedding_path, result_path, float_value, nlist_s, cache_size_s, linkage_value, dynamic_thread, jaccard_calculation, num_thread)
        # [변경된 부분]
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f'file name: {str(fname)}')
        print(f'error type: {str(exc_type)}')
        print(f'error msg: {str(e)}')
        print(f'line number: {str(exc_tb.tb_lineno)}')
        sys.exit(1)
