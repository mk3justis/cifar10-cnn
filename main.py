from data_loader import get_data_loaders
from model import CNN_model
from train import train_model
from utils import plot_loss_and_accuracy
import matplotlib.pyplot as plt
import torchvision.utils
import torch

train_loader, test_loader, validation_loader = get_data_loaders()
model = CNN_model()

# Train the model
training_loss, validation_loss, training_accuracy, validation_accuracy = train_model(model, train_loader, validation_loader)

# Plot loss and accuracy
plot_loss_and_accuracy(training_loss, validation_loss, training_accuracy, validation_accuracy)



# instantiate a model
model2 = CNN_model()

# set this model to evaluation mode 
model2.eval()

# load a random batch of test set images
for images, labels in train_loader:
    break

# show the images
plt.imshow(torchvision.utils.make_grid(images, nrow=8).permute(1,2,0))
plt.axis(False)
plt.show()

# print the ground truth class labels for these images
print(labels)

# compute model output
outputs2 = model2(images)

# print the predicted class labels for these images
_, predicted2 = torch.max(outputs2.data, 1)
print(predicted2)

# compute accuracy on each batch of test set
correct = 0
samples = 0
avg_accuracy = 0
for images, labels in test_loader :
    images, labels = images.to(), labels.to()
    outputs2 = model2(images)
    _, predicted2 = torch.max(outputs2.data, 1)
    correct += (predicted2 == labels).sum().item()
    samples += labels.size(0)
    accuracy = correct / samples
    avg_accuracy += accuracy

# print the average accuracy
avg_accuracy /= samples
print(avg_accuracy)