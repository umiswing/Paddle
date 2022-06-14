import spconv.pytorch as spconv
import random
import torch
import numpy as np

batch_size = 8
x = 41
y = 1600
z = 1408
# kernel size
kx = 3
ky = 3
kz = 3
in_channel = 4
out_channel = 16
nnz = 136000
paddings=(0,0,0)
strides = (1,1,1)
dilations = (1,1,1)

#batch_size = 1
#x = 4
#y = 4
#z = 4
#in_channel = 1
#out_channel = 1
#nnz = batch_size * x*y*z
features = []
indices = []
spatial_shape = []

for i in range(nnz):
    feature = []
    idx = []
    for j in range(in_channel):
        feature.append(random.uniform(-1,-0.0001) * random.choice([-1,1]))
    features.append(feature)

    idx.append(random.randrange(0,batch_size))
    idx.append(random.randrange(0,x))
    idx.append(random.randrange(0,y))
    idx.append(random.randrange(0,z))
    indices.append(idx)

spatial_shape = [x,y,z]

features = torch.tensor(features).cuda()
indices = torch.tensor(indices).int().cuda()
input = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
kernel = spconv.SparseConv3d(in_channel, out_channel, kernel_size=(kx,ky,kz), stride=strides, padding=paddings,dilation=dilations, bias=False)
kernel.weight = torch.nn.Parameter(kernel.weight.cuda())
output = kernel(input)
#print(kernel.weight.movedim(0,4).size())
#print(output.indices.size())
#print(output.features.size())

np.savetxt('features', features.reshape(-1).cpu().numpy())
np.savetxt('indices', indices.transpose(0,1).reshape(-1).cpu().numpy(), fmt='%d')
np.savetxt('kernel', kernel.weight.movedim(0,4).detach().reshape(-1).cpu().numpy())
np.savetxt('out_features', output.features.detach().reshape(-1).cpu().numpy())
np.savetxt('out_indices', output.indices.detach().transpose(0,1).reshape(-1).cpu().numpy(), fmt='%d')