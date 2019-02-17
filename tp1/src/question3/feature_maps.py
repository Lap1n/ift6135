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
    model.eval()
    with torch.no_grad():
        outputs = model(torch_img.unsqueeze(0))

    # depending on the size of outputs, save all feature\activation maps
    print("Debug -- number of layers in model : {} :".format(len(outputs)))
    for i,output in enumerate(outputs):
        if(i==0):
            pass
        else:
            output = output.squeeze()
            print("Debug -- lenght of features maps for layer {} : {} : ".format(i, output.shape[0]))
            for j,feat in enumerate(output):
                img = feat.numpy()
                plt.figure()
                plt.imshow(img)
                plt.title("Activation map layer {}, feature {}".format(i,j))
                plt.savefig(os.join(save_dir, "layer{}_feat{}.png".format(i,j)))
                        
    
def getConfusionMatrix(model, valid_loader, savedir):
    model.eval()
    y_prob = None
    y_true = None
    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Wrap tensors in Variables
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
        
            # Forward pass
            outputs = model(inputs)
            
            # append outputs and labels 
            if(y_prob is not None):
                y_prob = np.vstack((y_prob, outputs[0].numpy()))
                y_true = np.hstack(labels.numpy())
            else:
                y_prob = outputs[0].numpy()
                y_true = labels.numpy()
            
            y_pred = np.argmax(y_prob, axis=1)
            
    cm = confusion_matrix(y_true, y_pred, [0,1])
    heatmap = sns.heatmap(cm)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(savedir + "valid_cm.png")
    
    # get bady classified images with high confidence
    bad = np.nonzero(y_pred - y_true)
    bad_highconfidence = list()
    
    for idx in bad:
        if (y_prob[y_pred] > 0.80):
            bad_highconfidence.append(idx)

    # get ambiguous classifications
    ambiguous = np.where(y_pred[0] > 0.45 and y_pred[0] < 0.55)
    
    return (bad_highconfidence, list(ambiguous))
    

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
            
    img,label = dataset[11000]
    feat_dir = os.join(path, "features")
    os.mkdir(feat_dir)
    getFeatureMaps(img, model, feat_dir)
    bad_highconfidence, ambiguous = getConfusionMatrix(model, valid_loader, path)
