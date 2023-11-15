from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


### Prepare data

def dataloader_data(datasets):
    training_data = datasets(
    root="data",
    train=True,
    download=True,
    transform=ToTensor())

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor())
    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) ## trộn dữ liệu True
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)



