import random

import pytest
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import optim, nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from COVID19DatasetClass import COVID19Dataset
from CnnModels import CNN1, CNN2, ResNet, BasicBlock
from EarlyStopping import EarlyStopping
from efficientnet_pytorch import EfficientNet

# -------------- 2 Σύνολο δεδομένων --------------

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = COVID19Dataset(root_dir='COVID-19_Radiography_Dataset', transform=transform)

# 1. Display 25 random images from the dataset
random_indexes = random.sample(range(len(dataset)), 25)  # 25 random indexes
dataset.display_batch(random_indexes)

# 2. Count the number of images for each class + Display
class_counts = [0] * len(dataset.classes)

for idx in tqdm(range(len(dataset)), desc="Counting Images", unit="image"):
    _, label = dataset[idx]  # Get the label for each image
    class_counts[label] += 1

plt.bar(dataset.classes, class_counts)
plt.title('Number of Images per Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.show()


# -------------- 3 Συναρτήσεις εκπαίδευσης και δοκιμής --------------

def confusion_matrix(y, y_pred):
    # Infer num_classes from the unique classes in y and y_pred
    num_classes = len(dataset.classes)

    # Initialize the confusion matrix with zeros
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Populate the confusion matrix
    for true, pred in zip(y, y_pred):
        true = true.item()  # Convert true label to scalar
        pred = pred.item()  # Convert predicted label to scalar
        cm[true, pred] += 1  # Update confusion matrix

    return cm

# Example Usage:
# y = torch.tensor([0, 1, 2, 2, 0, 1, 2, 1])
# y_pred = torch.tensor([0, 0, 2, 2, 0, 1, 2, 2])
#
# # Get confusion matrix
# cm = confusion_matrix(y, y_pred)
#
# print("Confusion Matrix:")
# print(cm)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader, position=0, desc='Training', leave=False)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, dataloader, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    correct_predictions = 0
    total_predictions = 0
    # mse = 0
    # mae = 0
    num_classes = len(dataset.classes)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for X, y in tqdm(dataloader, position=0, desc='Testing', leave=False):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            if num_classes > 2:  # Multiclass classification
                pred = torch.argmax(pred, dim=1)
            else:  # Binary classification
                pred = (pred >= 0.5).int()
            # mse += torch.mean((y - pred) ** 2)
            # mae += torch.mean(torch.abs(y - pred))
            cm += confusion_matrix(y, pred)
            correct_predictions += (pred == y).sum().item()
            total_predictions += y.size(0)
    loss /= num_batches
    # rmse = torch.sqrt(mse) / num_batches
    # mae /= num_batches
    accuracy = correct_predictions / total_predictions
    # accuracy = accuracy_score(all_true_labels, all_pred_labels)
    # print(f"Avg loss: {loss:>8f}, RMSE: {rmse:>7f}, MAE: {mae:>7f} \n")

    return loss, accuracy, cm


# --------------  4 Απλό συνελικτικό δίκτυο --------------


def train_model(model, train_dl, val_dl, loss_fn, optimizer, epochs, device):
    train_loss = []
    val_loss = []
    early_stopping = EarlyStopping(patience=epochs)
    for i in range(epochs):
        train_one_epoch(model, train_dl, loss_fn, optimizer, device)

        # validation
        with torch.no_grad():
            # print(f'Training set: ', end='')
            epoch_train_loss, epoch_train_accuracy, epoch_train_cm = test(model, train_dl, loss_fn, device)
            # print(f'Validation set: ', end='')
            epoch_val_loss, epoch_val_accuracy, epoch_val_cm = test(model, val_dl, loss_fn, device)
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)

        print(f'\nEpoch: {i+1}')
        print(f'\nTraining set loss: {epoch_train_loss} \nAccuracy: {epoch_train_accuracy} \nConfusion matrix: \n{epoch_train_cm}')
        print(f'\nValidation set loss: {epoch_val_loss} \nAccuracy: {epoch_val_accuracy} \nConfusion matrix: \n{epoch_val_cm}')

        # early stopping
        early_stopping(epoch_train_loss, epoch_val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", i)
            break
    return train_loss, val_loss




device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

generator = torch.Generator().manual_seed(42)
train_ds, val_ds, test_ds = random_split(dataset, [0.6, 0.2, 0.2], generator=generator)


batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)



model = CNN1(num_classes=len(dataset.classes))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
loss_fn = nn.CrossEntropyLoss(reduction='mean')
epochs = 20

# --------------  5 Συνελικτικό δίκτυο μεγαλύτερου βάθους --------------

# model = CNN2(num_classes=len(dataset.classes))
# model = model.to(device)
#
# optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
# loss_fn = nn.CrossEntropyLoss(reduction='mean')
# epochs = 20

# --------------  6 Με χρήση προεκπαιδευμένου δικτύου  --------------

# model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 4)
# model = model.to(device)
#
# optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
# loss_fn = nn.CrossEntropyLoss(reduction='mean')
# epochs = 5


# --------------  6b Με χρήση προεκπαιδευμένου δικτύου (ως μηχανισμό εξαγωγής χαρακτηριστικών) --------------

# model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# for param in model.parameters():
#     param.requires_grad = False
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 4)
# model = model.to(device)
#
# optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
# loss_fn = nn.CrossEntropyLoss(reduction='mean')
# epochs = 5

# --------------  7 Προαιρετικά (Bonus): Συνελικτικό δίκτυο με παραλειπόμενες συνδέσεις --------------

# model = ResNet(BasicBlock, [1, 1, 1, 1])
# model = ResNet(BasicBlock, [2, 2, 2, 2])
# model = ResNet(BasicBlock, [3, 3, 3, 3])
# model = model.to(device)
#
# optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
# loss_fn = nn.CrossEntropyLoss(reduction='mean')
# epochs = 5




train_loss, val_loss = train_model(model, train_dl, val_dl, loss_fn, optimizer, epochs, device)
loss, accuracy, cm = test(model, test_dl, loss_fn, device)
print(f"\nTest set loss: {loss}, \nAccuracy: {accuracy}, \nConfusion matrix: \n{cm}")


