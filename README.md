# CALL
CALL: context-aware low-latency retrieval in disk-based vector databases, HiPC'25

1. call_gp.py: Dedicated cluster cache + query grouping
2. call_gp_prefetch_nonpipeline.py : Dedicated cluster cache + query grouping + prefetch let cluster cache wait until the clusters are available on the cache
3. call_gp_prefetch_pipeline.py: Dedicated cluster cache + query grouping + prefetch and second lookup overlapping
	1. ./run_limited_modified.sh 5 python3 call_gp_prefetch_pipeline_balanced.py hotpotqa 0.5 100 30 50 complete dynamic vector 8 fg        (prefetch clusters of first query in next group)
	2. ./run_limited_modified.sh 5 python3 call_gp_prefetch_pipeline_balanced.py hotpotqa 0.5 100 30 50 complete dynamic vector 8 balance   (prefetch clusters of all queries in nex group in balanced manner)
