
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

MASTER_ADDR=192.168.1.221   # laptop IP
MASTER_PORT=12355          # pick any open port

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  onemoretime.py 10 1 --batch_size 32
