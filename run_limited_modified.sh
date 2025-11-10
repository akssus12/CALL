#!/bin/bash
echo 3 > /proc/sys/vm/drop_caches  # To free pagecache, dentries and inodes
# 기본 실행 시간 (10분) 및 부동소수점 기본값 (0.3)
DURATION=10
FLOAT_VALUE=0.3

# 실행 시간(DURATION), Python 실행 파일, 스크립트, 데이터셋, 부동소수점 값 처리
if [ -n "$1" ]; then DURATION="$1"; shift; fi
if [ -n "$1" ]; then PYTHON_EXEC="$1"; shift; fi
if [ -n "$1" ]; then PYTHON_SCRIPT="$1"; shift; fi
if [ -n "$1" ]; then DATASET_NAME="$1"; shift; fi
if [ -n "$1" ]; then FLOAT_VALUE="$1"; shift; fi
if [ -n "$1" ]; then CLUSTER_SIZE="$1"; shift; fi
if [ -n "$1" ]; then N_LIST="$1"; shift; fi
if [ -n "$1" ]; then CACHE_SIZE="$1"; shift; fi
if [ -n "$1" ]; then LINKAGE_VALUE="$1"; shift; fi
if [ -n "$1" ]; then DYNAMIC_THREAD="$1"; shift; fi
if [ -n "$1" ]; then JACCARD_CALCULATION="$1"; shift; fi
if [ -n "$1" ]; then NUM_THREAD="$1"; shift; fi

# 필수 인자 확인
if [ -z "$PYTHON_EXEC" ] || [ -z "$PYTHON_SCRIPT" ]; then
    echo "❌ 사용법: $0 <duration> <python_interpreter> <python_script.py> [dataset_name] [float_value] [cluster_size] [nlist] [cache_size] [linkage_value] [dynamic_thread] [jaccard_cal] [num_thread]"
    exit 1
fi

# timeout 명령어로 제한된 시간 동안 실행
echo "⏳ $DURATION분 동안 실행: $PYTHON_EXEC $PYTHON_SCRIPT $DATASET_NAME $FLOAT_VALUE $CLUSTER_SIZE $N_LIST $CACHE_SIZE $LINKAGE_VALUE $DYNAMIC_THREAD $JACCARD_CALCULATION $NUM_THREAD" 
timeout "${DURATION}m" "$PYTHON_EXEC" "$PYTHON_SCRIPT" "$DATASET_NAME" "$FLOAT_VALUE" "$CLUSTER_SIZE" "$N_LIST" "$CACHE_SIZE" "$LINKAGE_VALUE" "$DYNAMIC_THREAD" "$JACCARD_CALCULATION" "$NUM_THREAD"

# 종료 메시지 출력
if [ $? -eq 124 ]; then
    echo "🚨 $DURATION분이 지나 실행이 강제 종료되었습니다."
else
    echo "✅ 실행이 정상적으로 종료되었습니다."
fi