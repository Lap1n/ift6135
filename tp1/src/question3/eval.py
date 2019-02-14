# -*- coding: utf-8 -*-
"""
Generate sample submission
TODO : turn this into script

"""
import csv
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import CNN
from dataset import TestDataset

import torch 

if __name__ == '__main__': 
    
    PATH = "models/VGG_small.pt"
    use_cuda = torch.cuda.is_available()
    
    # Assuming the model was saved on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create instance of the model
    model = CNN.SmallVGG()
    
    #load saved model
    # if on cpu
    if (use_cuda):
        model.load_state_dict(torch.load(PATH))
        model.to(device)
        
    else:
        model.load_state_dict(torch.load(PATH, map_location=device))

    
    # Create ToTensor and Normalize base transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
   
    # Get test data
    img_directory = "testset/test/"
    dataset = TestDataset(img_directory, transform=transform)
    
    classes = ('Cat', 'Dog')
    
    BATCH_SIZE = 1
    testloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=False)
    model.eval()
    # newline='' is for Windows fix that adds an extra carriage return at the
    # end of each line
    with open('submissions/submission.csv', 'w', newline='') as csvfile:
        print("id,label")
        writer = csv.writer(csvfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "label"])
        for i,images in enumerate(testloader):
            # Wrap tensors in Variables
            images = Variable(images).to(device)
            
            # Forward pass
            outputs = model(images)
        
            # Get classification
            pred = torch.argmax(outputs,1)
            result = classes[pred]
            writer.writerow([i+1, result])
            print("{},{}".format(i,result))
    
        

        
        
        
    
