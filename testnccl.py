import torch
import torch.distributed as dist
import os
if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12355"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
dist.init_process_group("nccl", rank=0, world_size=1)
print("NCCL is working!")