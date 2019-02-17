# -*- coding: utf-8 -*-
import time
import torch
import torch.optim as optim
from torch.autograd import Variable


def getClassificationError(outputs, labels):
    # Return the classification error and number of predictions
    pred = torch.argmax(outputs, 1)
    error = torch.sum(torch.abs(labels - pred))
    return (error.item(), len(labels))


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

    # loss and accuracy for plotting
    train_loss = []
    valid_loss = []
    train_error = []
    valid_error = []

    # loop for n_epochs
    for epoch in range(n_epochs):

        # Average loss and accuracy per epoch
        total_train_loss = 0.0
        total_train_error = 0.0
        nb_train = 0.0

        # Running loss and accuracy (just for printing)
        running_loss = 0.0
        running_error = 0.0
        # Set printing frequency
        print_every = int(len(train_loader) / 100)

        # put the model into training mode
        model.train()

        for i, data in enumerate(train_loader):
            (inputs, labels) = data

            # Wrap them in a Variable object
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            # Set the parameter gradints to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = model(inputs)
            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            # Keep track of loss and accuracy
            total_train_loss += batch_loss.item()
            running_loss += batch_loss.item()
            error, nb_pred = getClassificationError(outputs, labels)
            total_train_error += error
            running_error += error
            nb_train += nb_pred

            # print every 10th bach of and epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t avg_train_loss: {:.2f}, classification error {:.2f}%".format(epoch + 1,
                                                                                                       int(100 * (
                                                                                                                   i + 1) / n_batches),
                                                                                                       running_loss / print_every,
                                                                                                       running_error / print_every))

                # Reset running_loss and running_errors
                running_loss = 0.0
                running_error = 0.0

        # Get train_loss and classification train_error for this epoch
        train_loss.append(total_train_loss / nb_train)
        train_error.append(total_train_error / nb_train)

        # Do a pass on the validation set
        total_val_loss = 0
        total_val_error = 0
        nb_val = 0

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
                val_error, nb_pred = getClassificationError(val_outputs, labels)
                total_val_error += val_error
                nb_val += nb_pred

        # Save validation error and loss
        valid_loss.append(total_val_loss / nb_val)
        valid_error.append(total_val_error / nb_val)

        print("Validation loss = {:.2f}".format(total_val_loss / nb_val))
        print("Total training classification error = {:.2f}".format(total_train_error / nb_train))
        print("Total validation classification error = {:.2f}".format(total_val_error / nb_val))

    print("Training finished")

    # Plot training and validation loss and error
    print(train_loss)
    print(valid_loss)
    print(train_error)
    print(valid_error)
