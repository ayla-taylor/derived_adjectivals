import torch
import torch.nn as nn
from sklearn import metrics

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 768
hidden_size = 500
num_classes = 2
num_epochs = 5
batch_size = 32
learning_rate = 0.001


# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


labels = torch.load('labels.pt')
inputs = torch.load('embed.pt')

labels = labels.to(device)
inputs = inputs.to(device)

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# total_step = len(train_loader)
for epoch in range(num_epochs):
    for i in range(0, labels.shape[0], batch_size):
        batched_inputs = inputs[i: i + batch_size]
        batched_labels = labels[i: i + batch_size]
        # Move tensors to the configured device
        # inputs = inputs.reshape(-1, 28 * 28).to(device)
        # batched_labels = batched_labels.to(device)

        # Forward pass
        outputs = model(batched_inputs)
        loss = criterion(outputs, batched_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        # if (i + 1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         # images = images.reshape(-1, 28 * 28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Accuracy: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')


labels = torch.load('eval_labels.pt')
inputs = torch.load('eval_embed.pt')

labels = labels.to(device)
inputs = inputs.to(device)

with torch.no_grad():
    correct = 0
    total = 0
    pred_labels = []

    for i in range(0, labels.shape[0], batch_size):
        batched_inputs = inputs[i: i + batch_size]
        batched_labels = labels[i: i + batch_size]
        # images = images.reshape(-1, 28 * 28).to(device)
        outputs = model(batched_inputs)
        _, predicted = torch.max(outputs.data, 1)
        pred_labels.extend(predicted.data)
        total += batched_labels.size(0)
        correct += (predicted == batched_labels).sum().item()

    f1_score = metrics.f1_score(labels.data, pred_labels)
    print('f1: {} %'.format(100 * f1_score))
