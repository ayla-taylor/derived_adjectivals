
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 768
hidden_size = 500
num_classes = 2
num_epochs = 5
batch_size = 32
learning_rate = 0.001

model = torch.load('model.ckpt')
model.eval()

labels = torch.load('eval_labels.pt')
inputs = torch.load('eval_embed.pt')

labels = labels.to(device)
inputs = inputs.to(device)

with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, labels.shape[0], batch_size):
        batched_inputs = inputs[i: i + batch_size]
        batched_labels = labels[i: i + batch_size]
        # images = images.reshape(-1, 28 * 28).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy: {} %'.format(100 * correct / total))