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

exported_program: torch.export.ExportedProgram = torch.export.export(
    mod, args=(x,)
)

exported_program.graph.print_tabular()

