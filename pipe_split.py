# A note of pre-requisite:
# 1. register pipe_split op with pytorch aten first;
# 2. put op in _side_effectful_functions.
# https://github.com/pytorch/pytorch/compare/rzou/pipe_split?expand=1

import torch
import torch.nn as nn


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        torch.ops.pippy.pipe_split()
        x = self.fc3(x)
        return x


x = torch.randn(1)
m = M()

ep = torch.export.export(m, args=(x,))
print(ep)
ep.graph.print_tabular()

print("################################### Unflatten ##################################")
# Unflatten
unflattened = torch.export.unflatten(ep)
unflattened.graph.print_tabular()
print(unflattened)
for name, param in unflattened.named_parameters():
    print(f"{name}: {param.size()}")


print("################################### PiPPy ##################################")
import pippy
from pippy.IR import pipe_split, Pipe

pipe = Pipe.from_tracing(
    m,
    num_chunks=1,
    example_args=(x,),
)

print(pipe.split_gm)
pipe.split_gm.graph.print_tabular()

for name, param in pipe.named_parameters():
    print(f"{name}: {param.size()}")

