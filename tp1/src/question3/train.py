# -*- coding: utf-8 -*-
import time
import torch
import torch.optim as optim
from torch.autograd import Variable

def getClassificationError(outputs, labels):
    # Return the classification error in percentage
    pred = torch.argmax(outputs,1)
    error = torch.sum(torch.abs(labels - pred))
    return error.item()/len(labels)

def train(model, train_loader, valid_loader, batch_size, n_epochs, learning_rate):
    
    # Print all of the hyperparameters of the training iteration
    print("====== HYPERPARAMETERS ======")
    print("batch_size= ", batch_size)
    print("epochs= ", n_epochs)
    print("learning_rate= ", learning_rate)
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create loss
    loss = torch.nn.CrossEntropyLoss()
    
    # Create optimizer function -- not adam --VANILLA SGD
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    n_batches = len(train_loader)

    # Define the lists to store loss and classification error
    train_loss = []
    valid_loss = []
    train_error = []
    valid_error = []
    
    # loop for n_epochs
    for epoch in range(n_epochs):

        # Reset these variable at the beginning of each epoch
        start = time.time() 
        total_train_loss=0
        total_train_error = 0 # classification error
        print_every  = int(len(train_loader)/100)
        
        # put the model into training mode
        model.train()

        for i, data in enumerate(train_loader,0):
            (inputs, labels) = data
            
            # Wrap them in a Variable object
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            
            # Set the parameter gradints to zero
            optimizer.zero_grad()
            
            # Forward pass, backward pass, optimize
            outputs = model(inputs)
            losses = loss(outputs, labels)
            losses.backward()
            optimizer.step()
            
            # Print statistics
            running_loss = losses.item()
            total_train_loss += losses.item()
            # Classification error
            error = getClassificationError(outputs, labels)
            total_train_error +=error
            
            # print every 10th bach of and epoch
            #if (i+1) % (print_every + 1) == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.2f}, classification error {:.2f}%".format(epoch+1, 
                          int(100 * (i+1) / n_batches), running_loss, error*100))
                
                # Rest running_loss
                #running_loss = 0.0
                
        # at the end of one epoch, do a pass on the validation set
        total_val_loss = 0
        total_val_error = 0
        
        # Put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                # Wrap tensors in Variables
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
            
                # Forward pass
                val_outputs = model(inputs)
                val_losses = loss(val_outputs, labels)
                total_val_loss += val_losses.item()
            
                # Classification error
                val_error = getClassificationError(val_outputs, labels)
                print("Validation classification error : {}".format(val_error))
                total_val_error += val_error
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(valid_loader)))
        print("Total training classification error = {:.2f}".format(total_train_error/len(train_loader)))
        print("Total validation classification error = {:.2f}".format(total_val_error/len(valid_loader)))
    
    print("Training finished")
