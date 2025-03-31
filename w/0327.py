import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.xl = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.y = df.iloc[:, 2].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

def __len__(self):
    return self.length

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
)

def forward(self, x):
    x = self.layer(x)
    return x

train_dataset = CustomDataset("../perceptron.csv")
train_dataloader = DataLoade (train_dataset, batch_size=64, shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    cost = 0.0
for x, y in train_dataloader:
    x = x.to(device)
    y = y.to(device)

    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cost += loss
    cost = cost / len(train_dataloader)
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + l:4d}, Cost : {cost:.3f}")

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([
            [0,0]
            [0,1]
            [1,0]
            [1,1]
    ]).to(device)
    outputs = model(inputs)
    print("_________")
    print(outputs)
    print("outputs <=0.5")


