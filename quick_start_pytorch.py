import torch
from torch import nn
from torch.utils.data import DataLoader ## mau && labels
from torchvision import datasets ## su dung mot so thu vien data co san
from torchvision.transforms import ToTensor
 
## su dung datasets de dowload data co san
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

## set up batch_size sap du lieu traning_test dataset

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") ## số dữ liệu trong batch, số chiều , H , W
    print(f"Shape of y: {y.shape} {y.dtype}") ### số dữ liệu đầu ra, kiểu dl
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() ## làm phẳnng dữ liệu ảnh về dạng 1D ( N, C*H*W)
        ## đối với các mô hình mạng CNN thì không yêu cầu phải làm phẳng
        self.linear_relu_stack = nn.Sequential( ####định nghĩa xử lý tuần tự
            ## sử dụng các lớp Linear FCN
            nn.Linear(28*28, 512), ## đưa dữ liệu đầu vào có cùng chiều == đầu vào lớp Dense
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) ## đẩu ra softmax để biến đổi xác suât cho các đầu ra sao cho tổng xác suất =1
        return logits
model = NeuralNetwork().to(device)
print(model)
## Loss function, sử dụng trong bài toán phân loại với xác suất
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


## Define vòng training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

## thực hiện lệnh test
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


##thực hiện train và in lỗi test và độ chính xác 
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# ## Save model 
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

### sau khi huấn luyện thì có thể sử dụng lại để dự đoán
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

## sử dụng dự đoán trên mô hình đã có với data từ tập test
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')