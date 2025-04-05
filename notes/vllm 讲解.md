# EP00 大模型部署
## 常见模型和显存要求
- 模型占用 & 推理占用
    - xB -> 2x GB,  llama-8b -> 16GB, llama-1b -> 2GB
    - Quantization
    - KVCache:  [https://lmcache.ai/kv_cache_calculator.html](https://lmcache.ai/kv_cache_calculator.html)
    - 如何节省显存 

## vLLM 状态监测
- `/health`和 `/metrices`端口
- 核心 metrics 监控
    - num_waiting_requests > 0
    - num_runnings_requests
    - SLO related
        * time_to_first_token
        * decoding_throughput: 当前生成 每秒多少 token

## 负载均衡和路由算法
- 复杂均衡: 如何判断 overload
- 实战中路由算法实例
    - prefix-caching
    - Round-robin
    - Session-based
    - 最大前缀匹配



# EP01-vllm
[https://www.bilibili.com/video/BV1DpoWYeEeH/](https://www.bilibili.com/video/BV1DpoWYeEeH/)

vllm 代码库

## 模块
- Entrypoint (LLM, API server)
- Engine
- Scheduler
- KV cache manager
    - Paged Attention (Toread: LMCache, vLLM production stack)
- Evictor 
    - `prefix caching` <-- (what if prefix doesn't match? CacheBlend. What if prefix cache on another machine? KV cache sharing across nodes)
    - KV cache optimization
        * DeepSeek (MLA)
- Worker
- Model executor (Model runner)
- Modelling
- Attention backend
    - Flash attention

## 周边
- Preprocessing / Postprocessing (tokenizer, detokenizer, sampler, multimodel processor)
- Distributed
- `torch.compile`
- Observability
- Config
- Testing
- CI / CD
- Formatting

## 优化
- Speculative decoing
- Disaggregated prefilling
- Chunked prefill
- Cascade inference
- Prefix caching

# EP02-vllm 
[https://www.youtube.com/watch?v=W83Zgbg8SkE&list=PLJj_urhaf2_qxpg8A5-6xoMvMLBKQMTX1](https://www.youtube.com/watch?v=W83Zgbg8SkE&list=PLJj_urhaf2_qxpg8A5-6xoMvMLBKQMTX1)

## Distributed inference
### why distributed inference?
#### Communication device:
- NVLink: direct communication between GPUs
- Infinity Band: High-speed connection between nodes
- RDMA: Remote direct memory access
    - RDMA NIC
    - Software solution
    - Key advantage: bypass operating system / zero copy
    - RoCE

#### Communication library: `vllm/distributed/device_communicators`
- `PyNccl`: communication for NVIDIA
- `shared memory`: OS
- `custom allreduce`: A kernel just for all reduce operation
    - Before:
        * 0 machine: [0]
        * 1 machine: [1]
        * 2 machine: [2]
        * 3 machine: [3]
    - after: 
        * 0 machine: [0, 1, 2, 3]
        * 1 machine: [0, 1, 2, 3]
        * 2 machine: [0, 1, 2, 3]
        * 3 machine: [0, 1, 2, 3]
- `torch.distributed`: provide wide support to a list of communication library

##### GroupCoordinator
#### Algorithm-side
- [TP]
- `vllm/model_executor/models/llama.py`

### Pipeline  parallel
- Much less requirement to device --> device connection hardware
- cost: not improve latency. 
    - Tensor parallel: directly improve latency
- Algorithm-side
    - worker in charge of a subset of layers
    - `vllm/model_executor/models/llama.py`
    - `self.start_layer` --> `self.end_layer`
    - between workers: communicate intermediateTensor
    - `get_pp_group()`
    - `vllm/workder/model_runner.py`: search `get_pp_group()`
- Expert parallel & data parallel (advanced)
    - why expert paralel:
        * Mistral / Mixtral / Deepseek model: Mixture of Experts (Moe)
            - Only ofr linear layers
            - Normal model: all weights participant in compuation
            - MoE: expert as granularity, only a small subset of experts participatn the computation, this subset of expert may be different between request
        * Place different experts onto differents GPUs --> expert parallel 
        * Algorithm
            - Expert prallel
                - Shuffle (deepep communication kernel)
                - Forward
                - Shuffle back 
        * TP  is for attention, EP is for linear layers

### DP (data parallel)
- max tp << ep needed
- tp<# attention head
- basic linear layer "degree of parallism" >> basic attention layer tp "degree of parallism" , parallel request to raise attention "degree ofparallism"
- Difficult to implement in practice
    - request padding to avoid deadlock

### Types of distributed inference: TP / PP / EP / DP
### PD Disaggregation
## Prefix caching
## Speculative decoding
## Benchmarking


# EP03-PD disaggregation
[https://www.youtube.com/watch?v=ih6fcJnhoJI](https://www.youtube.com/watch?v=ih6fcJnhoJI)

## What's Prefill and Decode
- prefill : process input prompt, generate KV cache
- decode: generate tokens based on the KV cache

## Why PD disaggregation
- Prefill: attention -- N tokens QKV --> generate KV cache, takes a long time
- Decode: attention N KV, 1 Q --> generate a new token, very fast
- initial logic: prioritize prefill
- problem: prefill will stop other request's decode
- solution: PD disaggregation, chunked prefill

## Key problems in PD disaggregation
- How to transfer KV cache
    - 2 modes: pooling mode, p2p mode
    - library: LMCache, MoonCacke, NIXL
- How to extract (and inject) KV cache from (to) vLLM
    - connector API: `distributed.kv_connector.simple_connector.py`
    - called in `model_runner.py`
        * before model forward: try receive KV cache (inject KV cache into vLLM's paged memory)
        * model forward
        * after model forward: extract KV cache form vLLM's paged memory and send it to outside
- When to send the request to P and D node
    - first P then D
    - first D then P

