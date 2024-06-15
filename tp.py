import os
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, init_device_mesh
from torch.distributed._tensor.placement_types import _Partial
from torch.export._trace import _export

aten = torch.ops.aten

PROJECTION_OPS = [
    aten.linear.default,
]
ELEMENTWISE_OPS = [
    aten.relu.default,
    aten.add.Tensor
]

NRANKS = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)


torch.manual_seed(0)

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin0 = torch.nn.Linear(256, 1024)
        self.register_buffer("const", torch.ones(3, 64, 1024))
        self.relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(1024, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin0(x)
        x = x + self.const
        x = self.relu(x)
        x = self.lin1(x)
        return x


x = torch.randn(3, 64, 256, device=device)

mod = M().to(device)

# Export
print("export.export:")
exported_program: torch.export.ExportedProgram = _export(
    mod, args=(x,),
    pre_dispatch=False,
)
gm = exported_program._graph_module
print(exported_program)
gm.graph.print_tabular()
module = exported_program.module()
ref = module(x)
print(f"Running original... {ref.size()}")

# Check placeholders
for node in gm.graph.nodes:
    if node.op == "placeholder" or node.op == "get_attr":
        print(node, node.meta["val"])

# Distributed runtime

# Initialize device mesh
dist.init_process_group("nccl")
device_mesh = init_device_mesh("cuda", (NRANKS,))

# Tensor parallel: feeding sharded tensor to exported graph
xr = DTensor.from_local(x, device_mesh, [Replicate()])
p_lin0_weight = distribute_tensor(mod.lin0.weight, device_mesh, [Shard(0)])
p_lin0_bias = distribute_tensor(mod.lin0.bias, device_mesh, [Shard(0)])
p_lin1_weight = distribute_tensor(mod.lin1.weight, device_mesh, [Shard(1)])
p_lin1_bias = DTensor.from_local(mod.lin1.bias, device_mesh, [Replicate()])
b_const = distribute_tensor(mod.const, device_mesh, [Shard(-1)])

# Run
print("Running tensor parallel ...")
y_partial = gm(p_lin0_weight, p_lin0_bias, p_lin1_weight, p_lin1_bias, b_const, xr)[0]
y = y_partial.to_local()
dist.all_reduce(y)
torch.testing.assert_close(ref, y)
print("Passed!")




# Tensor parallel: graph
def pairwise_parallel(gm):
    # Initialize sharding spec.
    for node in gm.graph.nodes:
        node.meta.setdefault("shard_spec", None)

    addmm_nodes = []
    sharding_propagator = DTensor._op_dispatcher.sharding_propagator
    # schema = sharding_propagator.op_to_schema_info.get(
    #     torch._ops.OpOverload(aten.addmm.default), None
    # )
    # assert schema is not None
    for node in filter(lambda node: node.target == aten.addmm.default, gm.graph.nodes):
        b = node.args[0]
        x = node.args[1]
        at = node.args[2]
        if len(addmm_nodes) % 2 == 0:
            # First addmm
            x.meta["shard_spec"] = Replicate()
            at.meta["shard_spec"] = Shard(1)
            b.meta["shard_spec"] = Shard(0)
            node.meta["shard_spec"] = Shard(1)
            #DTensor._op_dispatcher.sharding_propagator
        else:
            # Second addmm
            x.meta["shard_spec"] = Shard(1)
            at.meta["shard_spec"] = Shard(0)
            b.meta["shard_spec"] = Replicate()
            node.meta["shard_spec"] = _Partial()
        addmm_nodes.append(node)

    # Propogate sharding spec

    # for node in gm.graph.nodes:
    #     print(f"{node}: {node.meta['shard_spec']}")

# Run
pairwise_parallel(gm)
