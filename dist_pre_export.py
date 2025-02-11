# torchrun --nproc-per-node 2 --standalone demo.py

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


def apply_tp(model, mesh):
    parallelize_module(model.net1, mesh, ColwiseParallel(), src_data_rank=None)
    parallelize_module(model.net2, mesh, RowwiseParallel(), src_data_rank=None)


def main():
    # Initialize distributed environment
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create distributed model
    d_hid = 1024
    model = MLPModule(d_hid)
    model = model.to(device)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    apply_tp(model, mesh)

    bs = 2
    x = torch.rand(bs, d_hid, device=device)

    # **************************************
    # We would export model here and hope it
    # would capture the model's collectives
    # **************************************

    # Real run
    y = model(x)
    y.wait()
    print(y.shape)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
