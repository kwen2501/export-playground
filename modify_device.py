import torch
import torch.export._trace

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin0 = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.conv(x)
        a = self.lin0(x)
        constant = torch.ones(1, 3, 256, 256, device=x.device)
        a.add_(constant)
        return self.maxpool(self.relu(a))


def modify_graph_device(
    gm: torch.fx.GraphModule,
    device: torch.device,
):
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if "device" in node.kwargs:
                node.update_kwarg("device", device)
                print(f"Changed device of Node {node.name}")
                modified = True
    if modified:
        gm.recompile()


example_args = (torch.randn(1, 3, 256, 256),)

mod = M()

print("################################### _export_to_torch_ir ##################################")
gm = torch.export._trace._export_to_torch_ir(
    mod, args=example_args
)

modify_graph_device(gm, torch.device('cuda'))
modify_graph_device(gm, torch.device('cuda'))
modify_graph_device(gm, torch.device('cuda'))
gm.graph.print_tabular()

gm.to("cuda")
real_input = torch.randn(1, 3, 256, 256, device='cuda')
gm(real_input)

