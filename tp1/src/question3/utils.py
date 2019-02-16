import argparse
import errno
import os
from datetime import datetime
import torch
import torchvision.transforms as transforms
import PIL
import pickle

import CNN

def createTransforms():
    # Note normalization is applied after augmentation by default in Dataset classes
    # Create Normalization and ToTensor transforms
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    # Create Data augmentation transforms
    rand_tsfm = transforms.RandomApply([transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                        transforms.ColorJitter(hue=0.1, saturation=0.1),
                                        transforms.RandomGrayscale(3),
                                        transforms.RandomAffine(30, (0.1, 0.1), (0.7,1.3))])
    
    augmentation = transforms.Compose([transforms.ToPILImage(), rand_tsfm])
    
    return (norm, augmentation)

def getDataLoaders(dataset, valid_proportion=0.2, batch_size=1):
    # Separate training and validation set
    valid_size = int(valid_proportion * len(dataset))
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    print("Train dataset length : {}".format(len(train_dataset)))
    print("Validation dataset length : {}".format(len(valid_dataset)))

    # Create Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    return (train_loader, valid_loader)

def createDateTimeFolder(source_dir="models/"):
    newdir = os.path.join(source_dir, datetime.now().strftime('%m%d_%H%M%S'))
    try: 
        os.makedirs(newdir)
    except OSError as e: 
        if e.errno != errno.EEXIST:
            raise
      
    return (newdir+'/')

def createFolder(newdir):
    try: 
        os.makedirs(newdir)
    except OSError as e: 
        if e.errno != errno.EEXIST:
            raise
      
    return (newdir+'/')

def runArgParser():
    parser = argparse.ArgumentParser(usage="Trains specified model and outputs results in models/date-time folder")

    parser.add_argument("--model", help="The name of the model class you want to use. (show options) Default is best model.",
                        type=str)
    parser.add_argument("--params", help="The parameters of the model, if required")
    parser.add_argument("--saved_path", help="The path to a previous model from which to start training",
                        type=str)
    parser.add_argument("--outdir", help="Specify the output directory. By default we create a date-time directory in models folder",
                        type=str)
    parser.add_argument("--show", help="Shows loss and error graphs. Default:False. Automatically saved in output directory",
                      type=bool)
    args = parser.parse_args()
    
    # Name of model, list all available classes in CNN.py
    model_name = args.model
    if model_name == "BaseCNN":
        model = CNN.BaseCNN()
    elif model_name == "SmallVGG":
        model = CNN.SmallVGG()
    elif model_name == "BigVGG":
        model = CNN.BigVGG()
    else:
        model = CNN.SmallVGG()
        model_name = "SmallVGG"
    
    # Set the output directory
    # Create a date-time folder in models folder to save model, model info and figures
    if (args.outdir):
        outdir = createFolder(args.outdir)
    else:
        outdir = createDateTimeFolder()
        
    # Get last saved state_dict for a model if specified (continued training)
    if (args.saved_path):
        saved_path = args.saved_path
    else:
        saved_path = None
        
    # Show graph or not, default is False
    if (args.show):
        show_graph = True
    else:
        show_graph = False
        
    # Save params in dictionary
    model_dict = {'model':model,
                  'model_name': model_name,
                  'outdir': outdir,
                  'saved_path':saved_path}
    
    # put everything in config structure   
    return (model_dict, show_graph)

def saveModelDict(model_dict, save_dir, save_name="model_dict.pkl"):
    with open(os.path.join(save_dir, save_name), 'wb') as f:
        pickle.dump(model_dict,f, pickle.HIGHEST_PROTOCOL)

def loadModelDict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
    
