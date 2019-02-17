# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import PIL

import dataset as ds
from train import train

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*padding)/stride) + 1
    return output

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class SmallVGG(torch.nn.Module):
    def __init__(self):
        super(SmallVGG, self).__init__()

        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # input channels=3, output channels = 18
        self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)).to(self.device)

        self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)).to(self.device)

        self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)).to(self.device)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True))
  
        # Classifier -- fully connected part
        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(256*8*8, 1024),
                torch.nn.ReLU(True),
                torch.nn.Linear(1024, 100),
                torch.nn.ReLU(True),
                torch.nn.Linear(100,2),
                torch.nn.Sigmoid())).to(self.device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x) + x
        x = self.layer5(x) + x


        x = x.view(-1, 256*8*8)
        x = self.classifier(x)

        return x, x1, x2, x3

class BigVGG(torch.nn.Module):
    def __init__(self):
        super(BigVGG, self).__init__()

        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # input channels=3, output channels = 18
        self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)).to(self.device)

        self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)).to(self.device)

        self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)).to(self.device)


        # Classifier -- fully connected part
        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512*8*8, 2048),
                torch.nn.ReLU(True),
                torch.nn.Linear(2048, 200),
                torch.nn.ReLU(True),
                torch.nn.Linear(200,2)).to(self.device)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = x3.view(-1, 512*8*8)
        x = self.classifier(x)

        return [x, x1, x2, x3]

class ConvNet2(torch.nn.Module):

    def __init__(self):
        super(ConvNet2, self).__init__()

        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # input channels=3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1).to(self.device)
        self.conv2 = torch.nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1).to(self.device)
        self.conv3 = torch.nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1).to(self.device)
        self.conv4 = torch.nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1).to(self.device)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0).to(self.device)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0).to(self.device)

        # fully connected from input to hidden layer
        self.fc1 = torch.nn.Linear(18*16*16, 64).to(self.device)
        self.fc2 = torch.nn.Linear(64,18).to(self.device)

        # fully connected from hidden layer to output : 2 classes
        self.fc3 = torch.nn.Linear(18,2).to(self.device)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = x.view(-1, 18*16*16)
        
        # Computes the activation of the first fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x    

class BaseCNN(torch.nn.Module):
    
    def __init__(self):
        super(BaseCNN, self).__init__()
        
        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # input channels=3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # fully connected from input to hidden layer
        self.fc1 = torch.nn.Linear(18*32*32, 64)
        
        # fully connected from hidden layer to output : 2 classes
        self.fc2 = torch.nn.Linear(64,2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Reshape data to input to the input layer of the neural net
        # Size changes from (18,32,32) to (1, 18*32*32)
        x = x.view(-1, 18*32*32)
        
        # Computes the activation of the first fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':  
    
    # random seed for reproducible results 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create ToTensor and Normalize base transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    # Create Data augmentation transforms
    augmentation = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(64,64),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                        transforms.ColorJitter(hue=.05, saturation=.05)])


    img_directory = "trainset/"
    dataset = ds.CatDogDataset(img_directory, transform=transform, augment=augmentation)
    # datasets = torch.utils.data.random_split(dataset, [1000, dataset.__len__()-1000])
    # dataset = datasets[0]
    classes = ('cat', 'dog')
    
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    print("Train dataset length : {}".format(len(train_dataset)))
    print("Validation dataset length : {}".format(len(valid_dataset)))

    # Create Data loaders
    BATCH_SIZE = 1
    # TODO add num_workers
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)

    print("Debug -- CNN.py --  valid_loader shape : {}".format(len(valid_loader)))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working on device : {}".format(device))


    baseCNN = SmallVGG().to(device)
    baseCNN.apply(init_weights)
    train(baseCNN, train_loader, valid_loader, batch_size=BATCH_SIZE, n_epochs=8, 
            learning_rate=0.001, momentum=0.9, weight_decay=0.1)
    
    torch.save(baseCNN.state_dict(), "models/VGG_big_aug.pt")
