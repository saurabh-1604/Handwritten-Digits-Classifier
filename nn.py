import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class SimpleNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)
   
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

model = SimpleNetwork()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.03)

# Training Model.
epochs = 10
losses = []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(trainloader)
    losses.append(average_loss)
    print(f"Epoch {e+1}/{epochs}, Training Loss: {average_loss:.4f}")

plt.figure(figsize=(10,5))
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()


correct_count, all_count = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        for i in range(len(labels)):
            img = images[i].view(1,784)
            logps = model(img)
            ps = torch.exp(logps)
            _, pred_label = torch.max(ps, 1)
            true_label = labels[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

print("Number of Images Tested =", all_count)
print("Model Accuracy =", (correct_count/all_count))

torch.save(model.state_dict(), 'trained_model.pth')
