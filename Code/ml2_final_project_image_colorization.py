import cv2

"""## Load dataset"""

import os
import numpy as np

DATA_DIR = os.getcwd()


data_path =  '../Data'
train_list = [f for f in os.listdir(data_path) if f[:-2] == "Train"]
lab_img = []
real_img = []
RESIZE = (256, 256)
for index1, path in enumerate(train_list):
  print("loading %s dataset:" % (path))
  for index2, image in enumerate([f for f in os.listdir(data_path+'/'+path)]):
    # print(data_path + '/'+ image)
    img = cv2.resize(cv2.imread(data_path + '/'+ path+ '/'+ image), RESIZE) # real image
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # grayscale img in L*a*b* color space format
    lab_img.append(g_img)
    real_img.append(img)
    if index2%100 == 0:
      print("%d images loaded" %(index2))
  print('----------------------------')

"""## DCGAN demo"""

#%matplotlib inline
import argparse
import os
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

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "/content/drive/My Drive/Colab Notebooks/ml2_final_project/Flickr1024 Dataset/"
# dataroot = os.listdir('/content/drive/My Drive/Colab Notebooks/ml2_final_project/Flickr1024 Dataset/Train_4')
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 2
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 128
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

gray_img = np.array([i[:,:,0] for i in lab_img])
ab_img = np.array([i[:,:,1:] for i in lab_img])

real_img = np.array(real_img)


# print(l_range.min()/255*100,'->',l_range.max()/255*100)
# print(a_range.min()-128,'->',a_range.max()-128)
# print(b_range.min()-128,'->',b_range.max()-128)



gray_img = gray_img.reshape(len(gray_img), 1, 256, 256)
ab_img = np.transpose(ab_img, (0,3,1,2))

real_img = np.transpose(real_img, (0,3,1,2))


# Create the dataloader
gray_img_Tensor = TensorDataset(torch.Tensor(gray_img))
real_img_Tensor = TensorDataset(torch.Tensor(real_img))

Gray_dataloader = DataLoader(gray_img_Tensor, batch_size=batch_size, shuffle=True, num_workers=workers)
Real_dataloader = DataLoader(real_img_Tensor, batch_size=batch_size, shuffle=True, num_workers=workers)

# Cite: https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform((x))

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

gray_img = gray_img/255
ab_img = ab_img/255

L_img_Tensor = torch.Tensor(gray_img) #L*
ab_img_Tensor = torch.Tensor(ab_img) #a*, b*
train_dataset = CustomTensorDataset(tensors=(L_img_Tensor, ab_img_Tensor))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

torch.Tensor(gray_img).shape

# Plot some training images
real_batch = next(iter(Gray_dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Gray scale Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

real_batch[0].shape

# Plot some training images
real_batch = next(iter(Real_dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
from functools import reduce
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# # Generator Code
# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 16),
#             nn.ReLU(True),
#             # state size. (ngf*16) x 4 x 4
#             nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 8 x 8
#             nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 16 x 16
#             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 32 x 32
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 64 x 64
#             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 128 x 128
#         )

#     def forward(self, input):
#         return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# netG.apply(weights_init)

# Print the model
# print(netG)

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 1
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
# netD = Discriminator(ngpu).to(device)

# Try pretrained model here
netD = models.resnet18(pretrained=False, num_classes=2)
netD.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
netD = netD.to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# netD.apply(weights_init)

# Print the model
# print(netD)

def save_weights(to_save_dict, epoch):
    torch.save(to_save_dict, 'colorize_gan_{}.pth.tar'.format(epoch))

# Loss Functions and Optimizers
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Try RMSprop optimizers for both G and D
# optimizerD = optim.RMSprop(netD.parameters(), lr=lr, alpha=0.99)
# optimizerG = optim.RMSprop(netG.parameters(), lr=lr, alpha=0.99)

pixel_loss_weights = 1000
g_every = 1 # every number of imgs to update parameters

# %%
# Commented out IPython magic to ensure Python compatibility.
for epoch in range(num_epochs):
    for i, (y, uv) in enumerate(train_dataloader):
      # Adversarial ground truths
      valid = Variable(torch.Tensor(y.size(0), 1).fill_(1.0),
                      requires_grad=False).to(device)
      fake = Variable(torch.Tensor(y.size(0), 1).fill_(0.0),
                      requires_grad=False).to(device)

      yvar = Variable(y).to(device)
      uvvar = Variable(uv).to(device)
      real_imgs = torch.cat([yvar, uvvar], dim=1)
      print(yvar.shape)
      break
      optimizerG.zero_grad()
      uvgen = netG(yvar)
      # Generate a batch of images
      gen_imgs = torch.cat([yvar.detach(), uvgen], dim=1)

      # Loss measures generator's ability to fool the discriminator
      g_loss_gan = criterion(netD(gen_imgs), valid)
      g_loss = g_loss_gan + pixel_loss_weights * torch.mean((uvvar - uvgen)**2)

      if i % g_every == 0:
        g_loss.backward()
        optimizerG.step()

      optimizerD.zero_grad()
      # Measure discriminator's ability to classify real from generated samples
      real_loss = criterion(netD(real_imgs), valid)
      fake_loss = criterion(netD(gen_imgs.detach()), fake)
      d_loss = (real_loss + fake_loss) / 2
      d_loss.backward()
      optimizerD.step()
      if i % 100 == 0:
        print("Epoch: %d, iter: %d, [D loss: %f] [G total loss: %f] [G GAN Loss: %f]" 
              % (epoch, i, d_loss.item(), g_loss.item(), g_loss_gan.item()))
        
        save_weights({'D': netD.state_dict(), 'G': netG.state_dict(), 'epoch': epoch}, epoch)

