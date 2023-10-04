import matplotlib.pyplot as plt
import torch

def plot_loss_and_accuracy(training_loss, validation_loss, training_accuracy, validation_accuracy, num_epochs=64) :
    # plot the training and validation loss
    plt.plot(range(num_epochs), training_loss[:num_epochs], label='Training Loss')
    plt.plot(range(num_epochs), validation_loss[:num_epochs], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    # plot the training and validation accuracy
    plt.plot(range(num_epochs), training_accuracy[:num_epochs], label='Training Accuracy')
    plt.plot(range(num_epochs), validation_accuracy[:num_epochs], label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()