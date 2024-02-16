import torch

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.lin0 = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        x = self.conv(x)
        a = self.lin0(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))


with torch.device("meta"):
    example_args = (torch.randn(1, 3, 256, 256),)
    example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

    mod = M()

print("################################### Export ##################################")
exported_program: torch.export.ExportedProgram = torch.export.export(
    mod, args=example_args, kwargs=example_kwargs
)
exported_program.graph.print_tabular()

print("################################### Unflatten ##################################")
# Unflatten
unflattened = torch.export.unflatten(exported_program)
unflattened.graph.print_tabular()
print(unflattened)

print("################################### _export_to_torch_ir ##################################")
gm = torch.export._trace._export_to_torch_ir(
    mod, args=example_args, kwargs=example_kwargs
)
gm.graph.print_tabular()

