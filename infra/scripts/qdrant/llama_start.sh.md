```shell
#!/bin/bash
set -eo pipefail

# run_qdrant.sh

cd $HOME/qdrant
$HOME/qdrant/qdrant --config-path $HOME/qdrant/config.yaml --uri http://0.0.0.0:6333

```

**$HOME/qdrant/config.yaml**
```yaml

# Qdrant Configuration for Intel N100 (bee node)
# Optimized for 16GB RAM + NVMe + 4-core Gracemont E-cores

storage:
  storage_path: ./storage
  on_disk_payload: true
  
  optimizers:
    max_optimization_threads: 1
    indexing_threshold_kb: 20000
    flush_interval_sec: 5
    default_vacuum_min_interval: 86400
    throttle_when_hot: true
    
  performance:
    direct_io: true
    max_read_ops_per_sec: 8000
    read_prefer_sequential: true
  
  mmap:
    preallocate: true
    lock_ram: false

vectors:
  default:
    size: 768
    distance: Cosine
    on_disk: true

hnsw_index:
  on_disk: true
  m: 8
  ef_construct: 64
  full_scan_threshold: 10000

quantization_config:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: false

performance:
  max_search_threads: 2
  simd: false
  prefer_cpu_power_save: true
  
  cache:
    payload_cache_size_mb: 4096
    vector_cache_size_mb: 2048
    chunk_size_bytes: 8192

service:
  max_request_size_mb: 32
  http_port: 6333
  grpc_port: 6334
  host: 0.0.0.0
  
  network:
    tcp_buffer_mb: 4
    keep_alive_sec: 300
    tcp_nodelay: true

cluster:
  enabled: false

log_level: INFO


```