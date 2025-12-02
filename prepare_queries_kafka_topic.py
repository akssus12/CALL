import json
import time
import numpy as np
import os
from multiprocessing import Process
from kafka import KafkaProducer
from beir import util

def download_and_load_beir_queries(dataset_name="scifact"):
    """
    BEIR 데이터셋을 다운로드하고 queries.jsonl에서 text 필드만 추출하여 리스트로 반환합니다.
    """
    # 1. BEIR 데이터셋 다운로드 경로 설정
#    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
    out_dir = "datasets"
    
    print(f"Downloading BEIR dataset: {dataset_name}...")
    # 데이터셋 다운로드 및 압축 해제 (이미 존재하면 건너뜀)
    data_path = util.download_and_unzip(url, out_dir)
    
    # queries.jsonl 파일 경로
    queries_path = os.path.join(data_path, "queries.jsonl")
    
    all_texts = []
    
    # 2. queries.jsonl 파일 파싱
    if os.path.exists(queries_path):
        print(f"Loading queries from {queries_path}...")
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 'text' 필드 추출
                    if 'text' in data:
                        all_texts.append(data['text'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {queries_path}")
    else:
        print(f"File not found: {queries_path}")

    print(f"Loaded {len(all_texts)} queries.")
    return all_texts

def producer(texts):
    """
    Kafka로 텍스트 데이터를 전송합니다.
    """
    # Kafka Producer 설정
    producer = KafkaProducer(bootstrap_servers=['163.239.199.205:9092'])
    topic = "msmacro"
    
    print(f"Kafka producer started - Sending {len(texts)} items to topic '{topic}'")
    
    try:
        # 추출된 텍스트들을 순차적으로 전송 (무한 루프가 필요하면 while True로 감싸거나 texts를 순회)
        # 여기서는 데이터를 한번 쭉 보내는 것으로 작성하되, 
        # 기존 코드처럼 시뮬레이션을 위해 리스트를 반복하고 싶다면 아래 주석을 해제하세요.
        
        # while True: # 무한 반복이 필요한 경우
            for sentence in texts:
                # 데이터 전송
                producer.send(topic, sentence.encode('utf-8'))
                print(f"Sent: {sentence[:50]}...") # 로그가 너무 길지 않게 잘라서 출력
                
                # Poisson 분포를 이용한 대기 시간 (기존 로직 유지)
                interval = np.random.poisson(1)
                if interval < 0:
                    interval = 0
                #time.sleep(interval)
                
    except Exception as e:
        print(f"Error in producer: {e}")
    finally:
        producer.close()
        print("Kafka producer closed.")

if __name__ == "__main__":
    # 1. BEIR 데이터셋 다운로드 및 쿼리 로드
    # 원하는 데이터셋 이름으로 변경 가능 (예: 'scifact', 'nfcorpus', 'fiqa' 등)
    # HotpotQA 데이터셋이 BEIR에 포함되어 있으므로 'hotpotqa'를 사용할 수도 있습니다.
    target_dataset = "msmacro" 
    
    queries = download_and_load_beir_queries(target_dataset)

    if queries:
        # 2. 프로세스 시작 (데이터가 있을 경우에만)
        p = Process(target=producer, args=(queries,))
        p.start()
        p.join() # 메인 프로세스가 자식 프로세스 종료를 기다림
    else:
        print("No queries loaded to send.")
