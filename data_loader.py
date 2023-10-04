import torchvision.transforms as transforms
import torch.utils.data as data

import torchvision.datasets as datasets

def get_data_loaders(batch_size=64) :
    transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_set = datasets.CIFAR10(root='/content/data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='/content/data', train=False, download=True, transform=transform)

    train_set, validation_set = data.random_split(dataset=train_set, lengths=[.8, .2])

    # write dataloader for train set
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # write dataloader for test set
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # write dataloader for validation set
    validation_loader = data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, validation_loader