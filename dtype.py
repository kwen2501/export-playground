import torch

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # an Embedding module containing 10 tensors of size 3
        self.embedding = torch.nn.Embedding(10, 3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embedding(x)
        z = torch.zeros_like(y)
        return z


x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

mod = M()
mod.to(torch.bfloat16)

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

