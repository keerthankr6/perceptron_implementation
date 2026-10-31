import torch
import torch.nn as nn
torch.manual_seed(42)

# inputs
input = torch.tensor([1.0, 2.0, 3.0])

neuron = nn.Linear(in_features = 3, out_features = 1)

output = neuron(input)

#verify ..formula for linear layer is correct

print(output == neuron.weight @ input + neuron.bias)  # True
print(output)