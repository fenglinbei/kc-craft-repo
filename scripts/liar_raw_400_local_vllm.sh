unset LD_PRELOAD
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset TORCH_DISTRIBUTED_DEBUG
unset NCCL_ASYNC_ERROR_HANDLING
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset NCCL_ALGO NCCL_PROTO
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_LEVEL=LOC

python scripts/run_pipeline.py \
  --config configs/liar_raw_local_vllm.yaml  \
  --mode full \
  --input data/liar_raw_400_test.jsonl \
  --output outputs/liar_raw_400_test_output.jsonl