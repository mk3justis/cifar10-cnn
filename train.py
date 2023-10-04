import torch
import torch.optim as optim
import torch.nn as nn
from model import CNN_model
from data_loader import get_data_loaders

model = CNN_model()

CEL = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.01)

def train_model(model, train_loader, validation_loader, num_epochs=100) :
    # Initialize empty lists to store training loss, training accuracy, validation loss, validation accuracy 
    # Will use these lists to plot the loss history.
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    # Loop through the number of epochs
    for epoch in range(num_epochs) :
        # set model to train mode
        model.train()

        # iterate over the training data in batches
        for images, labels in train_loader :
            
            # get the image inputs and labels from current batch
            images, labels = images.to(), labels.to()
            
            # set the optimizer gradients to zero to avoid accumulation of gradients
            optimizer.zero_grad()

            # compute the output of the model
            outputs = model(images)

            # compute the loss on current batch
            loss = CEL(outputs, labels)
            
            # backpropagate the loss
            loss.backward()

            # update the optimizer parameters
            optimizer.step()

            # update the train loss and accuracy
            training_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == labels).sum().item() / labels.size(0)
            training_accuracy.append(train_accuracy)
            

        # compute the average training loss and accuracy and store in respective arrays
        avg_training_loss = sum(training_loss[-len(train_loader):]) / len(train_loader)
        avg_training_accuracy = sum(training_accuracy[-len(train_loader):]) / len(train_loader)

        # set the model to evaluation mode
        model.eval()

        # keeping the gradient computation turned off, loop over batches of data from validation set.
        with torch.no_grad() :
            for images, labels in validation_loader :
                images, labels = images.to(), labels.to()
                # compute output of the model
                outputs = model(images)

                # compute the loss
                loss = CEL(outputs, labels)
                
                # compute the validation loss and accuracy
                validation_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                validate_accuracy = (predicted == labels).sum().item() / labels.size(0)
                validation_accuracy.append(validate_accuracy)

        # compute the average validation loss and accuracy and store in respective lists
        avg_validation_loss = sum(validation_loss[-len(validation_loader):]) / len(validation_loader)
        avg_validation_accuracy = sum(validation_accuracy[-len(validation_loader):]) / len(validation_loader)

        # print the training loss, training accuracy, validation loss and validation accuracy at the end of each epoch
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Training Loss: {avg_training_loss:.5f}, Training Accuracy: {avg_training_accuracy:.5f}")
        print(f"Validation Loss: {avg_validation_loss:.5f}, Validation Accuracy: {avg_validation_accuracy:.5f}")

        # save the model parameters once in every 5 epochs
        if (epoch + 1) % 5 == 0 :
            torch.save(model.state_dict(), f"model_{epoch+1}.pth")