import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


from dataset import CatDogDataset
from utils import getDataLoaders, runArgParser
from utils import createTransforms, saveModelDict
import CNN
import train

if __name__ == '__main__':
    
    model, model_dict, show_graph = runArgParser()
    
    # Print model details
    print("Training model : {}".format(model_dict['model_name']))
    print("Saving in : {}".format(model_dict['outdir']))
    
    # random seed for reproducible results
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create ToTensor and Normalize base transforms
    normalization, augmentation = createTransforms()

    # Get the dataset and dataloader
    dataset = CatDogDataset("trainset/", transform=normalization, augment=augmentation)
    train_loader, valid_loader = getDataLoaders(dataset, valid_proportion=0.2, batch_size=1)

    # Get device
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working on device : {}".format(device))
    
    # Output directory to save figures and models
    outdir = model_dict['outdir']
    
    # get saved model parameters if there are some
    saved_path = model_dict['saved_path']
    if(saved_path):
        if(use_cuda):
            model.load_state_dict(torch.load(saved_path))
        else:
            model.load_state_dict(torch.load(saved_path, map_location=device))
            
    
    # Get other parameters      
    batch_size = model_dict['batch_size']
    n_epochs = model_dict['n_epochs']
    learning_rate = model_dict['learning_Rate']
    
    model.to(device)
    model.apply(CNN.init_weights)
    output = train.train(model, train_loader, valid_loader, batch_size=batch_size,
                         n_epochs=n_epochs,learning_rate=learning_rate, outdir=outdir)
    
    train_loss, train_error, valid_loss, valid_error = output
    
    print("Losses and errors")
    print(train_loss)
    print(train_error)
    print(valid_loss)
    print(valid_error)

    # Add output to model_dict
    model_dict['train_loss' ] =  train_loss
    model_dict['train_error'] =  train_error
    model_dict['valid_loss' ] =  valid_loss
    model_dict['valid_error'] =  valid_error
    
    # Output directory to save figures and model
    outdir = model_dict['outdir']

    # Save model_dict
    print(model_dict)
    saveModelDict(model_dict, outdir)
    
    # plot and save figure
    # TODO, do not display if not in desktop environment
    # loss
    plt.figure()
    plt.plot(train_loss, label="Training")
    plt.plot(valid_loss, label="Validation")
    plt.title("Training and Validation cross entropy loss")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(outdir + "loss.png")
    
    # error
    plt.figure()
    plt.plot(train_error, label="Training")
    plt.plot(valid_error, label="Validation")
    plt.title("Training and Validation classification error (%)")
    plt.xlabel("number of epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(outdir + "error.png")
    
    
    torch.save(model.state_dict(), outdir + "model.pt")
