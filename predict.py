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
from skimage import io

def visualization(test_ds, preds):
    print("############# Randomly visualizing some images with the predicted classes")
    idx = random.sample(range(len(test_ds)), 9)
    images = []
    rows = 3
    columns = 3
    fig = plt.figure(figsize=(10, 7))
    for i in range(len(idx)):
        images.append(io.imread(test_ds.val_images[idx[i]]))
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i+1)
        # showing image
        plt.imshow(images[i])
        plt.axis('off')
        label = preds[idx[i]]
        plt.title(test_ds.class_names[label])
    plt.show()



def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    preds = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            preds.append(pred.argmax(1))
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return preds

def prediction(dataset_path=None, weight_path=None, batch_size=1):
    print("############### Running the Test ############################")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_ds = UCMerced(dataset_path=dataset_path, device=device, is_train=False)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = torch.load(weight_path)
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    preds = test_loop(test_dataloader, model=model, loss_fn=criterion)
    visualization(test_ds, preds)



if __name__ == "__main__" :


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to load the test dataset')
    parser.add_argument('--weight_path', type=str, help='Path to load the weights')
    args = parser.parse_args()

    prediction(dataset_path=args.dataset_path, weight_path=args.weight_path)