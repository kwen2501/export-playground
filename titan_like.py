import torch

# Simple module for demonstration
class layer(torch.nn.Module):
    def forward(self, x, const) -> torch.Tensor:
        return x + const

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.ones(64, 256))
        self.layers = torch.nn.ModuleList([layer() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.const)
        return x


# Export
mod = M()
x = torch.randn(64, 256)
ep = torch.export.export(mod, (x,))
unflattened = torch.export.unflatten(ep)

# Run
try:
    unflattened(x)
except Exception as e:
    print(e)

# Inspection
def print_submod(root, fqn_):
    mod_itr_ = root.get_submodule(fqn_)
    print(f"{fqn_}: {mod_itr_.graph}")

print_submod(unflattened, "")
print_submod(unflattened, "layers.0")
