import os
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import random
import matplotlib
# Use non-interactive backend if no display is available
if not os.environ.get('DISPLAY', ''):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import multiprocessing
import time

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets

import subprocess
import shutil

DATA_PATH = os.path.join('./data/')

# Auto-download the dataset if not exists
if not os.path.exists(DATA_PATH) or not os.path.exists(os.path.join(DATA_PATH, 'bangla')):
    print("Dataset not found. Downloading Bangla-Money-Dataset from GitHub...")
    os.makedirs(DATA_PATH, exist_ok=True)
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/nsojib/Bangla-Money-Dataset.git",
        os.path.join(DATA_PATH, "bangla")
    ], check=True)
    git_dir = os.path.join(DATA_PATH, "bangla", ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir)
    print("Dataset downloaded and extracted successfully!")

print(os.listdir(DATA_PATH))

# use gpu if you have
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print ("MPS device is available. Successfully initiated:")
    print (x)
    device = mps_device
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is available.")
    print("GPU device count:", torch.cuda.device_count())
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = torch.device("cpu")

print("Use device:",device)

# set random seed
SEED = 6220
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
multiprocessing.set_start_method("fork")
#----------------------------------------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.485, 0.229)
    ])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.485, 0.229)
    ])
#----------------------------------------------------------------------------------------------------------
# Specify the path to the training folder.
TRAINDATA_PATH = os.path.join('.','data','bangla', 'Training')
# print(TRAINDATA_PATH)
label_name = {"1": 0, "2": 1, "5": 2, "10": 3, "20": 4, "50": 5, "100": 6, "500": 7, "1000": 8}

def get_img_info(data_dir):
        imgpath = []
        imglabel = []
        for root, dirs, _ in os.walk(data_dir):
            # print("root: ", root)
            # print("dirs: ", dirs)
            # Traverse categories
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # Traverse images
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = label_name[sub_dir]
                    imgpath.append(path_img)
                    imglabel.append(int(label))

        # Return image paths and labels in data_info
        return imgpath,  imglabel

imgpath,  imglabel = get_img_info(TRAINDATA_PATH)
#----------------------------------------------------------------------------------------------------------
class Custom_dataset(Dataset):
    def __init__(self,trainData,trainLabel,transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.images = trainData
        self.label = trainLabel
        self.transform = transform

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = self.images[index]
        img = Image.open(imgpath).convert('RGB')

        label = self.label[index]
        if self.transform:
          img = self.transform(img)

        return img, label

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.images)
#----------------------------------------------------------------------------------------------------------
# Spilt the train and valid data
train_img, val_img, train_label, val_label = train_test_split(imgpath, imglabel, test_size=0.2, random_state=42)

train_set = Custom_dataset(train_img, train_label, transform=train_transform)
valid_set = Custom_dataset(val_img, val_label, transform=valid_transform)

print('Number of total training data:', len(train_set))
print('Number of total validation data:', len(valid_set))

class_num = 9
classes = ('1', '10', '100', '1000', '2', '20', '5', '50', '500')

# Loaded Datasets to DataLoaders
# please change the batch_size
trainloader = DataLoader(train_set, batch_size=16 , shuffle=True, num_workers = 0)
validloader = DataLoader(valid_set, batch_size=16 , shuffle=False, num_workers = 0)
#----------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Checking the dataset
for images, labels in trainloader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

for img,labels in trainloader:
    # load a batch from train data
    break

# this converts it from GPU to CPU and selects first image
img = img.cpu().numpy()[0]
#convert image back to Height,Width,Channels
img = np.transpose(img, (1,2,0))
#show the image
plt.imshow(img)
plt.show()

for images, labels in validloader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

for img_test,labels in validloader:
    # load a batch from train data
    break

# this converts it from GPU to CPU and selects first image
img_test = img_test.cpu().numpy()[0]
#convert image back to Height,Width,Channels
img_test = np.transpose(img_test, (1,2,0))
#show the image
plt.imshow(img_test)
plt.show()
#----------------------------------------------------------------------------------------------------------
##############################################
# Build your model here!
#
# Practice:
#   Try to implement VGG-16 with pytorch !
##############################################

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        ##############################################################################
        #                          TODO: implement VGG-16.                           #
        ##############################################################################
        self.conv_block = nn.Sequential(
#---------------------------------------------------------------------------------------------------------------------------
            nn.Conv2d(3,  64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
#---------------------------------------------------------------------------------------------------------------------------
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
#---------------------------------------------------------------------------------------------------------------------------
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
#---------------------------------------------------------------------------------------------------------------------------
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
#---------------------------------------------------------------------------------------------------------------------------
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )
#---------------------------------------------------------------------------------------------------------------------------
        self.feat_classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, class_num)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.feat_classifier(x)
        return x

model = VGG16()
model.to(device)
#---------------------------------------------------------------------------------------------------------------------------
print("device: ",device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 1

model.train()
#---------------------------------------------------------------------------------------------------------------------------
# Training model
train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []

# Specify the saving weight path
SAVING_PATH = './Saving_Path'
os.makedirs(SAVING_PATH, exist_ok=True)

valid_loss_min = np.inf # track change in validation loss

for epoch in range(1, epochs+1):# loop over the dataset multiple times

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('running epoch: {}'.format(epoch))

    # train the model
    model.train()
    train_correct = 0
    train_total = 0
    for data, target in tqdm(trainloader):
      # move tensors to GPU if CUDA is available
      data, target = data.to(device), target.to(device)
      # clear the gradients of all optimized variables
      optimizer.zero_grad()
      # forward pass: compute predicted outputs by passing inputs to the model
      output = model(data)
      # calculate the batch loss
      loss = criterion(output, target)
      # backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()
      # perform a single optimization step (parameter update)
      optimizer.step()
      # update training loss
      train_loss += loss.item()*data.size(0)
      # update training Accuracy
      train_total += target.size(0)
      _, predicted = torch.max(output.data, 1)
      train_correct += (predicted == target).sum().item()


    # validate the model
    model.eval()
    valid_correct = 0
    valid_total = 0
    for data, target in tqdm(validloader):
        # move tensors to GPU if CUDA is available
        target = target.long()
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)
        # update validation Accuracy
        valid_total += target.size(0)
        _, predicted = torch.max(output.data, 1)
        valid_correct += (predicted == target).sum().item()
    # calculate average losses
    train_loss = train_loss/len(trainloader.dataset)
    valid_loss = valid_loss/len(validloader.dataset)

    # print training/validation statistics
    print('Training Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(train_loss,(100 * train_correct / train_total)))
    print('Validation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(valid_loss,(100 * valid_correct / valid_total)))

    train_loss_list.append(train_loss)
    train_acc_list.append(100 * train_correct / train_total)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(100 * valid_correct / valid_total)

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), SAVING_PATH+'/model_weight.pth')
        valid_loss_min = valid_loss

print('Finished Training')
#---------------------------------------------------------------------------------------------------------------------------
def plt_loss_acc(list_to_draw,name):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    if name=="train_loss":
      ax1.set_title('Train Loss')
      ax1.plot(list_to_draw)
    elif name=="train_acc":
      ax1.set_title('Train Accuracy')
      ax1.plot(list_to_draw)
    elif name=="valid_loss":
      ax1.set_title('Valid Loss')
      ax1.plot(list_to_draw)
    elif name=="valid_acc":
      ax1.set_title('Valid Accuracy')
      ax1.plot(list_to_draw)

    ax1.set_xlabel('epoch')
    plt.show()

def plt_loss_acc_all():
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('All acc and loss')

    ax1.plot(train_loss_list)
    ax1.plot(train_acc_list)
    ax1.plot(valid_loss_list)
    ax1.plot(valid_acc_list)

    ax1.legend(['train_loss', 'train_acc', 'valid_loss', 'valid_acc'], loc='upper left')
    ax1.set_xlabel('epoch')
    plt.show()

def plt_acc_all():
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('All acc')

    ax1.plot(train_acc_list)
    ax1.plot(valid_acc_list)

    ax1.legend(['train_acc', 'valid_acc'], loc='upper left')
    ax1.set_xlabel('epoch')
    plt.show()

def plt_loss_all():
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('All loss')

    ax1.plot(train_loss_list)
    ax1.plot(valid_loss_list)

    ax1.legend(['train_loss', 'valid_loss'], loc='upper left')
    ax1.set_xlabel('epoch')
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------
plt_loss_acc(train_loss_list, "train_loss")
plt_loss_acc(train_acc_list, "train_acc")
plt_loss_acc(valid_loss_list, "valid_loss")
plt_loss_acc(valid_acc_list, "valid_acc")
plt_loss_all()
plt_acc_all()
#---------------------------------------------------------------------------------------------------------------------------
print(predicted[0].item())
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as data_utils
TESTDATA_PATH = './data/bangla/Testing'
for data in os.walk(TESTDATA_PATH):
  test_data = [f for f in data[2] if f.endswith('.jpg')]
test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225])
    ])
class Custom_testset(Dataset):
    def __init__(self,testData,transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.images = testData
        #self.label = trainLabel
        self.transform = transform

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = self.images[index]
        img = Image.open(imgpath).convert('RGB')

        if self.transform:
          img = self.transform(img)

        return img

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.images)

def trueclass(num):
  if num==0:
    return 1
  elif num==1:
    return 2
  elif num==2:
    return 5
  elif num==3:
    return 10
  elif num==4:
    return 20
  elif num==5:
    return 50
  elif num==6:
    return 100
  elif num==7:
    return 500
  elif num==8:
    return 1000

imgpath=[]
prediction=[]
for photo in test_data:
  path_img = os.path.join(TESTDATA_PATH,photo)
  imgpath.append(path_img)
# print('id = ',test_data)
test_set = Custom_testset(imgpath,test_transform)
testloader = DataLoader(test_set, batch_size=1 , shuffle=False, num_workers = 0)
for images in testloader:
    print('Image batch dimensions:', images.shape)
    break

#images = np.transpose(images, (1,2,0))
for images in testloader:
  #print('image = ',images)
  images=images.to(device)
  output = model(images)
  predicted = torch.argmax(output,dim=1)
  #print('Image predicted label = :', predicted.item())
  prediction=np.append(prediction,trueclass(predicted.item()))

example={'image':test_data,
      'class':prediction}
df = pd.DataFrame(example)
print(df)
df.to_csv('./data/example.csv',index=False)