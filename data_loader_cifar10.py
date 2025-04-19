import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

def get_cifar10_loaders(root='./data',
                        batch_size=128,
                        image_size=224,
                        num_workers=5):
    """
    Returns PyTorch DataLoader objects for the CIFAR-10 dataset.

    Args:
        root (str): Root directory for the dataset.
        batch_size (int): Batch size for train/test DataLoader.
        image_size (int): The size to which input images are resized (e.g., 224 for ViT).
        num_workers (int): Number of worker processes for data loading.

    Returns:
        trainloader (DataLoader): DataLoader for training set.
        testloader (DataLoader): DataLoader for test set.
    """

    # Mean and std of CIFAR-10 for normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    task = 'multi-class'

    # Transforms for training
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Transforms for testing
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Datasets
    trainset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform_test
    )

    # Dataloaders
    trainloader = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    testloader = data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    NUM_CLASSES = 10
    for x, y in trainloader:
        print("Shape of input batch:", x.shape)
        break

    return trainloader, testloader,testloader, NUM_CLASSES, task