import torch

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin0 = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.lin0(x))


example_args = (torch.randn(1, 3, 256, 256),)

mod = M()

# Export
print("export.export:")
exported_program: torch.export.ExportedProgram = torch.export.export(
    mod, args=example_args,
)
print(exported_program)
exported_program.graph.print_tabular()

# torch._export._export_to_torch_ir
print("_export_to_torch_ir:")
gm = torch.export._trace._export_to_torch_ir(
    mod, args=example_args,
)
print(gm)
gm.graph.print_tabular()

# unflattened
unflattened_module = torch.export.unflatten(exported_program)
print("unflattened_module:")
print(unflattened_module)
unflattened_module.graph.print_tabular()
for name, param in unflattened_module.named_parameters():
    print(f"{name}, {param.size()}")

