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
def draw_colorized_image(test_img_floder, number_of_img_shown, epoch, save_path, origial_size=False, random_img=False):
    gpu = 0
    device = torch.device("cpu")
    model = 'colorize_gan_{}.pth.tar'.format(epoch-1)
    G = Generator(gpu).to(device)
    G.load_state_dict(torch.load(model,map_location={'cuda:0': 'cpu'})['G'])

    # test_img_path = '../Data/Test/001_L.png'
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

        if origial_size:
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
draw_colorized_image('../Data/Test', 2, 300, 'example1')

# %%
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