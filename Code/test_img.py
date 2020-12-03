"""
Simple demo for colorization a gray scaled image

"""
import torch
from torch.autograd import Variable
from scipy.ndimage import zoom
import cv2
import os
from PIL import Image
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd import Variable
from functools import reduce

# %%
# Generator Code
class shave_block(nn.Module):
    def __init__(self, s):
        super(shave_block, self).__init__()
        self.s=s
    def forward(self,x):
        return x[:,:,self.s:-self.s,self.s:-self.s]

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential( # Sequential,
            nn.ReflectionPad2d((40, 40, 40, 40)),
            nn.Conv2d(1,32,(9, 9),(1, 1),(4, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        ),
                    shave_block(2),
                    ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
                ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        ),
                    shave_block(2),
                    ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
                ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        ),
                    shave_block(2),
                    ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
                ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        ),
                    shave_block(2),
                    ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
                ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.BatchNorm2d(128),
                        ),
                    shave_block(2),
                    ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
                ),
            nn.ConvTranspose2d(128,64,(3, 3),(2, 2),(1, 1),(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,(3, 3),(2, 2),(1, 1),(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,2,(9, 9),(1, 1),(4, 4)),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)

# %%
gpu = 0
device = torch.device("cpu")
model = 'colorize_gan_99.pth.tar'

G = Generator(gpu).to(device)

G.load_state_dict(torch.load(model,map_location={'cuda:0': 'cpu'})['G'])

test_img_path = '../Data/Train_1/001_L.png'
test_img = cv2.cvtColor(cv2.resize(cv2.imread(test_img_path), (256, 256)), cv2.COLOR_BGR2LAB)/255
test_img_L = test_img[..., 0].reshape(1, 1,256,256)
img_variable = Variable(torch.Tensor(test_img_L))

ab_gen = G(img_variable)
ab = ab_gen.cpu().detach().numpy()
ab = ab*255

# %%
gen_lab_img = np.transpose(np.vstack((test_img_L[0,...]*255, ab[0,...])), (1, 2, 0))
gen_lab_img = gen_lab_img.astype(np.uint8)
# %%
print(gen_lab_img[...,0].min(), '->', gen_lab_img[...,0].max())
print(gen_lab_img[...,1].min(), '->', gen_lab_img[...,1].max())
print(gen_lab_img[...,2].min(), '->', gen_lab_img[...,2].max())

# %%
gen_img = cv2.cvtColor(gen_lab_img, cv2.COLOR_LAB2BGR)
plt.imshow(gen_img)
plt.show()

# %%
plt.imshow(cv2.resize(cv2.imread(test_img_path), (256, 256)))
plt.show()