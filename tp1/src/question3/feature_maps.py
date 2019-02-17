"""
Script to extract feature maps and confusion matrix for validation data
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision.transforms as transforms

from utils import createTransforms, getDataLoaders
from dataset import CatDogDataset
import CNN
import os

def getFeatureMaps(torch_img, model, save_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        torch_img = torch_img.unsqueeze(0)
        torch_img = torch_img.to(device)
        outputs = model(torch_img)

    # depending on the size of outputs, save all feature\activation maps
    print("Debug -- number of layers in model : {} :".format(len(outputs)))
    for i,output in enumerate(outputs):
        if(i==0):
            pass
        else:
            output = output.squeeze()
            print("Debug -- lenght of features maps for layer {} : {} : ".format(i, output.shape[0]))
            for j,feat in enumerate(output):
                img = feat.cpu()
                img = img.numpy()
                plt.figure()
                plt.imshow(img)
                plt.title("Activation map layer {}, feature {}".format(i,j))
                plt.savefig(os.path.join(save_dir, "layer{}_feat{}.png".format(i,j)))
                        
    

if __name__ == '__main__':

    # Get scripts arguments
    parser = argparse.ArgumentParser(usage="Get Activation maps and Confusion matrix for validation set of specified model")

    parser.add_argument("dir", help="The directory containing the model and model_dict",
                        type=str)
    
    parser.add_argument("model_name", help="The name of the model",
                        type=str)
    parser.add_argument("model_file", help="Name of the file containing the saved model (ex : best_model.pt)")

    args = parser.parse_args()
    
    path = args.dir
    model_name = args.model_name
    model_file = args.model_file

# random seed for reproducible results
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create ToTensor and Normalize base transforms
    normalization, augmentation = createTransforms()

    # Get the dataset and dataloader
    dataset = CatDogDataset("trainset/", transform=normalization, augment=augmentation)
    train_loader, valid_loader = getDataLoaders(dataset, valid_proportion=0.2, batch_size=8)

    # Get device
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working on device : {}".format(device))
    
    
    # Create instance of model 
    if (model_name == "SmallVGG"):
        model = CNN.SmallVGG()
    else:
        print("This model is not implemented")
        
    # Load saved model
    model_path = os.path.join(path, model_file)
    
    if(use_cuda):
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
            
    img,label = dataset[10213]
    img.to(device)
    
    feat_dir = os.path.join(path, "features")
    if not(os.path.isdir(feat_dir)):
        os.mkdir(feat_dir)
    getFeatureMaps(img, model, feat_dir)
