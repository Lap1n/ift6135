# -*- coding: utf-8 -*-
"""
Generate sample submission
TODO : turn this into script

"""
import csv
from torch.autograd import Variable
import torchvision.transforms as transforms

import CNN
from Dataset import TestDataset

import torch 

if __name__ == '__main__': 
    
    PATH = "models/best_model.pt"
    
    # Assuming the model was saved on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create instance of the model
    model = CNN.baseCNN()
    
    #load saved model
    # if on cpu
    if (device == "cpu"):
        model.load_state_dict(torch.load(PATH), map_location=device)
        
    else:
        model.load_state_dict(torch.load(PATH))
        model.to(device)

    
    # Create ToTensor and Normalize base transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
   
    # Get test data
    img_directory = "testset/test"
    dataset = TestDataset(img_directory, transform=transform)
    
    classes = ('Cat', 'Dog')
    
    BATCH_SIZE = 64
    testloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=False)
    model.eval()
    with open('submissions/submission.csv', 'rb') as csvfile:
        print("id,label")
        for i,images in enumerate(testloader,1):
            # Wrap tensors in Variables
            images = Variable(images).to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get classification
            pred = torch.argmax(outputs,1)
            result = classes[pred]
            print(i,result)
    
        

        
        
        
    