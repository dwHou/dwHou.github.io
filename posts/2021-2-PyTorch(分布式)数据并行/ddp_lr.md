```python
from torch import nn
import torch

import torch
from torch.autograd import Function
from model.MPRNet import MPRNet
from loss import L1_Charbonnier_loss, MultiSupervision
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
model = MPRNet().cuda()

# print(model.state_dict().keys())

loss = L1_Charbonnier_loss()

input = torch.randn(4, 3, 3, 112, 112, requires_grad=True).cuda()
info = torch.randn(4, 1, 112, 112, requires_grad=True).cuda()
target = torch.randn(4, 3, 112, 112).cuda()


# Simulate DP
b1 = input[0:2]
b2 = input[2:4]
i1 = info[0:2]
i2 = info[2:4]

b1 = model(b1, i1)
b2 = model(b2, i2)

output = loss(torch.cat((b1[0], b2[0]), 0), target)

output.backward()

grad1 = model.ffnet.inc.double_conv[0].weight.grad[0][0][0]

# grad1 = model.stage3_orsnet.orb3.fusion1.headConv.conv.weight.grad[0][0][0]
# model.stage3_orsnet.orb3.fusion1.headConv.conv.weight.grad = None
model.ffnet.inc.double_conv[0].weight.grad = None
# ffnet.inc.double_conv.0.weight


# Simulate DDP
b1 = input[0:2]
b2 = input[2:4]
i1 = info[0:2]
i2 = info[2:4]
t1 = target[0:2]
t2 = target[2:4]

b1 = model(b1, i1)
output = 0.5 * loss(b1[0], t1) # 1 / world_size
output.backward()

b2 = model(b2, i2)
output = 0.5 * loss(b2[0], t2)
print(output.item())
output.backward()



print(model.stage3_orsnet.orb3.fusion1.headConv.conv.weight.grad[0][0][0])

# grad2 = model.stage3_orsnet.orb3.fusion1.headConv.conv.weight.grad[0][0][0]
grad2 = model.ffnet.inc.double_conv[0].weight.grad[0][0][0]

print (grad1 / grad2)
```

