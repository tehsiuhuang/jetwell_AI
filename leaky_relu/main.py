import time
import torch
import torch.nn.functional as F
from torch.fx import symbolic_trace, GraphModule
from leaky_relu_ext import leaky_relu  # 你自己寫的 op
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import allow_in_graph

class SimpleNetWithResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(784, 256)
        self.fc1 = nn.Linear(256, 256)

        self.conv = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(256, 256)  # residual shortcut
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = x.view(x.size(0), 16, -1)
        residual = self.fc2(x.view(x.size(0), -1))

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)

        x = x + residual
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def replace_relu_with_leaky_relu(model, slope=0.1):
    model.eval()
    traced = symbolic_trace(model)

    new_graph = torch.fx.Graph()
    env = {}

    for node in traced.graph.nodes:
        if node.op == 'placeholder':
            env[node.name] = new_graph.placeholder(node.name)

        elif node.op == 'call_function' and node.target == F.relu:
            input_node = env[node.args[0].name]
            new_node = new_graph.call_function(leaky_relu, (input_node, slope))
            env[node.name] = new_node

        elif node.op == 'call_module':
            submod = dict(model.named_modules())[node.target]
            if isinstance(submod, torch.nn.ReLU):
                input_node = env[node.args[0].name]
                new_node = new_graph.call_function(leaky_relu, (input_node, slope))
                env[node.name] = new_node
            else:
                args = torch.fx.map_arg(node.args, lambda n: env[n.name])
                env[node.name] = new_graph.call_module(node.target, args)

        elif node.op == 'call_method':
            args = torch.fx.map_arg(node.args, lambda n: env[n.name])
            env[node.name] = new_graph.call_method(node.target, args)

        elif node.op == 'output':
            new_graph.output(env[node.args[0].name])

        else:
            args = torch.fx.map_arg(node.args, lambda n: env[n.name])
            env[node.name] = new_graph.node_copy(node, lambda n: env[n.name])

    return GraphModule(model, new_graph)

def benchmark(model, x, name="patched", repeat=1000, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    print(f"[{name}] Device: {x.device}, Avg time: {(end - start)/repeat*1000:.3f} ms")

# === Run ===
model = SimpleNetWithResidual()
patched = replace_relu_with_leaky_relu(model, slope=0.05)

x_cpu = torch.randn(256, 784)
x_gpu = x_cpu.cuda()

benchmark(patched.cpu(), x_cpu, "leaky_relu_cpu")
benchmark(patched.cuda(), x_gpu, "leaky_relu_gpu")



benchmark(torch.compile(patched, backend="inductor").cpu(), x_cpu, "AOTI leaky_relu_cpu")
benchmark(torch.compile(patched, backend="inductor").cuda(), x_gpu, "AOTI leaky_relu_gpu")

