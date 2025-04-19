import os
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
import medmnist
from medmnist import INFO, Evaluator
import wandb
from collections import Counter
from torchvision.utils import make_grid

def tmp_func(x):
    return x.convert('RGB')

def log_class_distribution(loader, class_names):
    """Logs the distribution of classes in the dataset to W&B."""
    counts = torch.zeros(len(class_names), dtype=torch.long)  # Use long dtype for bincount

    for _, labels in loader:
        # Ensure labels are on CPU and are of type long
        labels = labels.cpu().long()
        
        # Remove any extra dimensions (e.g., from [128, 1] to [128])
        if labels.dim() > 1:
            labels = labels.squeeze()

        # Ensure labels are non-negative integers
        if not torch.all(labels >= 0):
            raise ValueError("Labels contain negative values.")

        counts += torch.bincount(labels, minlength=len(class_names))

    total = counts.sum().item()
    percentages = (counts / total) * 100

    # Convert class_names to list if it's a dictionary
    if isinstance(class_names, dict):
        class_names_list = [class_names[str(i)] for i in range(len(class_names))]
    else:
        class_names_list = class_names

    # Debug statements
    print("Class names:", class_names_list)
    print("Counts:", counts)
    print("Percentages:", percentages)

    table = wandb.Table(columns=["Class", "Count", "Percentage"])
    for i in range(len(class_names_list)):
        table.add_data(class_names_list[i], counts[i].item(), percentages[i].item())

    # Log table to W&B
    wandb.log({"class_distribution_table": table})


def log_sample_images(loader, n_samples=16):
    """Logs sample images from the dataset to W&B."""
    images, labels = next(iter(loader))
    grid = make_grid(images[:n_samples], nrow=4, normalize=True)
    wandb.log({"sample_images": wandb.Image(grid, caption="Sample Images")})

# Load Data
def get_loader(args):
    data_flag = args.dataset  # expected [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist]
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    class_names = info['label']

    # Transforms for train
    train_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.Lambda(tmp_func),
        torchvision.transforms.AugMix(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = DataClass(split='train', transform=train_transform, download=True, size=224)

    # Transforms for val
    val_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.Lambda(tmp_func), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    val_dataset = DataClass(split='val', transform=val_transform, download=True, size=224)

    # Transforms for test
    test_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.Lambda(tmp_func), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    test_dataset = DataClass(split='test', transform=test_transform, download=True, size=224)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               drop_last=True) #pin_memory=True if args.is_cuda else False
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.n_workers,
                                              drop_last=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.n_workers,
                                              drop_last=False)
    
    print('<<<<<<<<<<<<<<<<<<<<<<<< Dataset Information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(train_dataset)
    print("===================")
    print(val_dataset)
    print('<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(test_dataset)
    print('<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # Log data insights to W&B
    log_class_distribution(train_loader, class_names)
    log_sample_images(train_loader)

    return train_loader, val_loader, test_loader, task, n_classes
