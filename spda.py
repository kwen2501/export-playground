import torch
import torch.nn.functional as F

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # an Embedding module containing 10 tensors of size 3
        self.embedding = torch.nn.Embedding(10, 3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = xk = xv = self.embedding(x)
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        return out


x = torch.LongTensor([1, 2, 4, 5])

mod = M()

exported_program: torch.export.ExportedProgram = torch.export.export(
    mod, args=(x,)
)

exported_program.graph.print_tabular()

exported_program.graph_module.print_readable()

for node in exported_program.graph_module.graph.nodes:
    try:
        val = node.meta["val"]
        dtype = val.dtype
    except:
        val = None
        dtype = None
    print(node.name, val, dtype)

pre_dispatch_ir = torch._export.capture_pre_autograd_graph(
    mod, args=(x,)
)

pre_dispatch_ir.print_readable()
pre_dispatch_ir.graph.print_tabular()

