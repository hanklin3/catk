# %%
# https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.pool.radius.html#torch_geometric.nn.pool.radius

import torch
from torch_geometric.nn import radius

x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
batch_x = torch.tensor([0, 0, 0, 0])
y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
batch_y = torch.tensor([0, 0])
assign_index = radius(x, y, 1.5, batch_x, batch_y)
print(assign_index)
# %%
