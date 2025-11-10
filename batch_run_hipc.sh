#!/bin/bash

########## hotpotqa ##########
# nlist 30 
#./run_limited_modified.sh 5 python3 cagr_rag_only_grouping_disk_top10.py hotpotqa 0.4 100 30 50 complete dynamic vector 8
#./run_limited_modified.sh 5 python3 cagr_rag_grouping_and_prefetch_disk_nomerge_executor_hipc.py hotpotqa 0.5 100 30 50 complete dynamic vector 8
#./run_limited_modified.sh 5 python3 cagr_rag_grouping_and_prefetch_disk_nomerge_executor_hipc.py hotpotqa 0.5 100 30 50 complete dynamic vector 8

./run_limited_modified.sh 5 python3 call_gp_prefetch_pipeline_balanced.py hotpotqa 0.5 100 30 50 complete dynamic vector 8 fg  # fg(first query of next group) or balance(most clusters on next group)



########## fever ##########
# nlist 30 
#./run_limited_modified.sh 10 python3 cagr_rag_optimized_hipc.py fever 0.4 100 30 50 complete dynamic vector 8

########## nq ##########
# nlist 30 
#./run_limited_modified.sh 10 python3 cagr_rag_optimized_hipc.py nq 0.4 100 30 50 complete dynamic vector 8
