from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(batch_size=64, num_workers=2):
    # Data augmentation and normalization for training
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # Just normalization for validation/test
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # Optimize for Windows and CUDA by using pin_memory and persistent_workers if num_workers > 0
    kwargs = {"pin_memory": True}
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs
    )

    return trainloader, testloader
