"""
Simple demo for colorization a gray scaled image

"""
import cv2
import os
import torch
from random import sample
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import matplotlib.pyplot as plt
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
# test_img_floder = '../Data/customize_test'
# test_img_path = [img_path for img_path in os.listdir(test_img_floder)]
# print(test_img_path)
# %%
def draw_colorized_image(test_img_floder, number_of_img_shown, epoch, save_path, original_size=False, random_img=False):
    """
    Input:

    test_img_folder -> The folder where the images in
    number_of_img_shown -> Show number of images
    epoch -> Use which epoch of model
    save_path -> file of save figure
    original_size=False -> Whether resize to original size
    random_img=False -> Whether random show images
    """

    """
    Output:

    Print out the figures contains grayscale, real image and colorized image
    Save the figure to the save_path
    """
    gpu = 0
    device = torch.device("cpu")
    model = 'colorize_gan_{}.pth.tar'.format(epoch-1)
    G = Generator(gpu).to(device)
    G.load_state_dict(torch.load(model,map_location={'cuda:0': 'cpu'})['G'])

    # test_img_path = '../Data/Test/002_L.png'

    if not random_img:
        test_img_path = [test_img_floder+'/'+img_path for img_path in os.listdir(test_img_floder)]
    else:
        test_img_path = sample([test_img_floder+'/'+img_path for img_path in os.listdir(test_img_floder)], number_of_img_shown)


    for row, img_path in enumerate(test_img_path):
        print(img_path)
        img = cv2.imread(img_path)
        size = (img.shape[1], img.shape[0])
        test_img = cv2.resize(img, (256, 256))
        test_img_lab = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
        test_img_lab_scaled = test_img_lab/255
        test_img_L = test_img_lab_scaled[..., 0].reshape(1, 1,256,256)
        img_variable = Variable(torch.Tensor(test_img_L))

        ab_gen = G(img_variable)
        ab = ab_gen.cpu().detach().numpy()
        ab = ab*255

        gen_lab_img = np.transpose(np.vstack((test_img_L[0,...]*255, ab[0,...])), (1, 2, 0))
        gen_lab_img = gen_lab_img.astype(np.uint8)

        # show test img
        gen_img = cv2.cvtColor(gen_lab_img, cv2.COLOR_LAB2RGB) # for plt show use RGB channel not BGR

        idx = row*3

        if original_size:
            plt.subplot(number_of_img_shown, 3, (idx+1))
            plt.title('Gray img')
            plt.imshow(cv2.resize(test_img_lab[..., 0], size), cmap='gray')

            plt.subplot(number_of_img_shown, 3, (idx+2))
            plt.title('Real img')
            plt.imshow(cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), size))

            plt.subplot(number_of_img_shown, 3, (idx+3))
            plt.title('Fake img')
            plt.imshow(cv2.resize(gen_img, size))
        else:
            plt.subplot(number_of_img_shown, 3, (idx+1))
            plt.title('Gray img')
            plt.imshow(test_img_lab[..., 0], cmap='gray')

            plt.subplot(number_of_img_shown, 3, (idx+2))
            plt.title('Real img')
            plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

            plt.subplot(number_of_img_shown, 3, (idx+3))
            plt.title('Fake img')
            plt.imshow(gen_img)

        if row+2 > number_of_img_shown:
            break

    plt.savefig(save_path + '_'+ str(epoch) + 'epochs' + '.png')
    plt.show()

# %%
draw_colorized_image('../Data/customize_test', 4, 100, 'example1')

# %% test for single image
gpu = 0
device = torch.device("cpu")
model = 'colorize_gan_{}.pth.tar'.format(500)
G = Generator(gpu).to(device)
G.load_state_dict(torch.load(model, map_location={'cuda:0': 'cpu'})['G'])

img_name = 'romanhed.jpg'
img_path = '../Data/customize_test/'+img_name
img = cv2.imread(img_path)
size = (img.shape[1], img.shape[0])
test_img = cv2.resize(img, (256, 256))
test_img_lab = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
test_img_lab_scaled = test_img_lab / 255
test_img_L = test_img_lab_scaled[..., 0].reshape(1, 1, 256, 256)
img_variable = Variable(torch.Tensor(test_img_L))

ab_gen = G(img_variable)
ab = ab_gen.cpu().detach().numpy()
ab = ab * 255

gen_lab_img = np.transpose(np.vstack((test_img_L[0, ...] * 255, ab[0, ...])), (1, 2, 0))
gen_lab_img = gen_lab_img.astype(np.uint8)

# show test img
gen_img = cv2.cvtColor(gen_lab_img, cv2.COLOR_LAB2RGB)  # for plt show use RGB channel not BGR

# plt.subplot(1, 3, 1)
# plt.title('Gray img')
# plt.imshow(cv2.resize(test_img_lab[..., 0], size), cmap='gray')
#
# plt.subplot(1, 3, 2)
# plt.title('Real img')
# plt.imshow(cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), size))

# plt.subplot(1, 3, 3)
plt.title('Fake img')
plt.imshow(cv2.resize(gen_img, size))

# %%
plt.savefig(img_name + '_' + str(500) + 'epochs' + '.png')

# %%
plt.show()
# %% Print out the structure of Generator
from torchviz import make_dot, make_dot_from_trace
from graphviz import Source

gpu = 0
device = torch.device("cpu")
model = 'colorize_gan_{}.pth.tar'.format(epoch - 1)
G = Generator(gpu).to(device)
G.load_state_dict(torch.load(model, map_location={'cuda:0': 'cpu'})['G'])
test_img_path = data_path + '/Test/001_L.png'
test_img = cv2.resize(cv2.imread(test_img_path), (256, 256))
test_img_lab = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
test_img_lab_scaled = test_img_lab / 255
test_img_L = test_img_lab_scaled[..., 0].reshape(1, 1, 256, 256)
img_variable = Variable(torch.Tensor(test_img_L))

ab_gen = G(img_variable)
model_arch = make_dot(G(img_variable), params=dict(G.named_parameters()))
Source(model_arch).render('structure')

# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fXdAV29_oMMpVhYtwAY4_qhM44qJZswL
"""

import os
import cv2
import sys
import math
import time
import glob
import random
import warnings
import datetime
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from google.colab import drive
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset

epoch = 0
n_epochs=100
dataset_name='test'
batch_size = 16
lr = 0.0005
b1 = 0.5
b2 = 0.999
decay_epoch = 100
n_cpu = 4
img_height = 256
img_width = 256
channels = 3
sample_interval = 100
checkpoint_interval = 338

warnings.filterwarnings('ignore')
DATA_DIR = os.getcwd()

class ImageDataset_color(Dataset):
  def __init__(self,root,transforms_=None,mode="train"):
    self.transform = transforms.Compose(transforms_)#transform
    self.files = sorted(glob.glob(root+"/*.*"))#dir
  def __getitem__(self,index):
    img_A = Image.fromarray(np.array(cv2.cvtColor(cv2.imread(self.files[index%len(self.files)]),cv2.COLOR_BGR2RGB)), "RGB")
    img_B = Image.fromarray(np.array(cv2.cvtColor(cv2.cvtColor(cv2.cvtColor(cv2.imread(self.files[index%len(self.files)]),cv2.COLOR_BGR2RGB),cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)), "RGB")
    img_A = self.transform(img_A)
    img_B = self.transform(img_B)       
    return {"A":img_A,"B":img_B}
  def __len__(self):
    return len(self.files)

transforms_=[transforms.Resize((256,256), Image.BICUBIC),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]


class UNetDown(nn.Module):
  def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
      super(UNetDown, self).__init__()
      layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
      if normalize:
          layers.append(nn.InstanceNorm2d(out_size))
      layers.append(nn.LeakyReLU(0.2))
      if dropout:
          layers.append(nn.Dropout(dropout))
      self.model = nn.Sequential(*layers)
  def forward(self, x):
      return self.model(x)
class UNetUp(nn.Module):
  def __init__(self, in_size, out_size, dropout=0.0):
      super(UNetUp, self).__init__()
      layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),nn.InstanceNorm2d(out_size),nn.ReLU(inplace=True),]
      if dropout:
          layers.append(nn.Dropout(dropout))
      self.model = nn.Sequential(*layers)
  def forward(self, x, skip_input):
      x = self.model(x)
      x = torch.cat((x, skip_input), 1)
      return x
class GeneratorUNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3):
      super(GeneratorUNet, self).__init__()
      self.down1 = UNetDown(in_channels, 64, normalize=False)
      self.down2 = UNetDown(64, 128)
      self.down3 = UNetDown(128, 256)
      self.down4 = UNetDown(256, 512, dropout=0.5)
      self.down5 = UNetDown(512, 512, dropout=0.5)
      self.down6 = UNetDown(512, 512, dropout=0.5)
      self.down7 = UNetDown(512, 512, dropout=0.5)
      self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
      self.up1 = UNetUp(512, 512, dropout=0.5)
      self.up2 = UNetUp(1024, 512, dropout=0.5)
      self.up3 = UNetUp(1024, 512, dropout=0.5)
      self.up4 = UNetUp(1024, 512, dropout=0.5)
      self.up5 = UNetUp(1024, 256)
      self.up6 = UNetUp(512, 128)
      self.up7 = UNetUp(256, 64)
      self.final = nn.Sequential(
          nn.Upsample(scale_factor=2),
          nn.ZeroPad2d((1, 0, 1, 0)),
          nn.Conv2d(128, out_channels, 4, padding=1),
          nn.Tanh(),)
  def forward(self, x):
      # U-Net generator with skip connections from encoder to decoder
      d1 = self.down1(x)
      d2 = self.down2(d1)
      d3 = self.down3(d2)
      d4 = self.down4(d3)
      d5 = self.down5(d4)
      d6 = self.down6(d5)
      d7 = self.down7(d6)
      d8 = self.down8(d7)
      u1 = self.up1(d8,d7)
      u2 = self.up2(u1,d6)
      u3 = self.up3(u2,d5)
      u4 = self.up4(u3,d4)
      u5 = self.up5(u4,d3)
      u6 = self.up6(u5,d2)
      u7 = self.up7(u6,d1)
      return self.final(u7)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

lambda_pixel = 100 #

# # Calculate output of image discriminator (PatchGAN)
patch = (1,16,16) #(1,16,16),img_height // 2 ** 4, img_width // 2 ** 4


gpu = 0
device = torch.device("cpu")
#model = 'colorize_gan_{}.pth.tar'.format(epoch - 1)
generator = GeneratorUNet()
generator = generator.cuda()
generator.load_state_dict(torch.load(DATA_DIR+"/generator_99.pth"))
#G = GeneratorUNet(gpu).to(device)
#generator.load_state_dict(torch.load(model, map_location={'cuda:0': 'cpu'})['G'])

test_path="../Data/Train_1"
dataloader_test=DataLoader(ImageDataset_color(test_path,transforms_=transforms_),batch_size=2,num_workers=4,)
#for epoch in range(epoch,n_epochs):

C=ImageDataset_color(test_path,transforms_=transforms_)

for i, batch in enumerate(dataloader_test):
  real_A = Variable(batch["B"].type(Tensor))
  fake_B = generator(real_A)

fake_B -=fake_B.min()
fake_B/=fake_B.max()
B=np.transpose(fake_B.cpu().detach().numpy(), (0,2,3,1))

plt.imshow(B[0,...])

