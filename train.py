
import os
from torch.utils.data import DataLoader, Dataset
import torch
import os
import argparse
import random
import matplotlib.pyplot as plt

from utils.dataset import UCMerced
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm



def test_loop(dataloader, model, loss_fn):
    print("############### Running Val Loop ############################")
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct



def save_figures(train, val) :
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_accuracy_graph.png')

def train_classification(dataset_path=None, n_epochs=10, batch_size=8, lr=0.001) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_ds = UCMerced(dataset_path=dataset_path, device=device)

    val_ds = UCMerced(dataset_path=dataset_path, device=device, is_train=False)



    train_dataloader = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)



    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, training_ds.num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    best_val_accuracy = 0.0

    train_losses = []
    val_accuracies = []

    for epoch in range(n_epochs) :
        
        for batch, (images, labels) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}', ncols=100, total=len(train_dataloader))):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(images)
                print(f"--------------------------->>> loss: {loss:>7f}")
                train_losses.append(loss)


        val_accuracy = test_loop(val_dataloader, model=model, loss_fn=criterion)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(model, fr'./weights/model_epoch{epoch}.pth')

    print("Done")
    save_figures(train_losses, val_accuracies)


if __name__ == "__main__" :


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()

    train_classification(dataset_path=args.dataset_path, n_epochs=50)
    