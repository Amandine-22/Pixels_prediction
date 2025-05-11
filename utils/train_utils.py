import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
from .data_utils import visualize_random_batch

def train(model, train_loader, optimizer, criterion, n_epochs, device): 
    model.train()
    model.to(device) 

    for e in range(n_epochs):
        current_loss = 0
 
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {e+1}/{n_epochs} [Train]")

        for i, data in progress_bar: 
            x_masked, x, _ = data
            x_masked = x_masked.to(device) 
            x = x.to(device)

            optimizer.zero_grad()

            output = model(x_masked)
            loss = criterion(output, x)

            loss.backward()
            optimizer.step()

            current_loss += loss.item()

            progress_bar.set_postfix(loss=current_loss / (i + 1))

        print(f"Epoch {e+1} finished, Average Loss: {current_loss / len(train_loader):.4f}")


def evaluate(model, test_loader, criterion, device, directory, visualize=True):
    model.eval()
    model.to(device) 
    total_loss = 0
    num_batch = 0

    with torch.no_grad(): 
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluation")

        for i, data in progress_bar: 
            num_batch += 1

            x_masked, x, _ = data
            x_masked = x_masked.to(device) 
            x = x.to(device)

            output = model(x_masked)
            loss = criterion(output, x) 

            total_loss += loss.item() 

            progress_bar.set_postfix(loss=total_loss / num_batch)


    average_loss = total_loss / num_batch
    print(f"Evaluation finished, Average Loss: {average_loss:.4f}")
    
    if visualize:
        visualize_random_batch(loader=test_loader, model=model, directory=directory)
        
    return average_loss
