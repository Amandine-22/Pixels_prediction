import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset, Dataset

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL.Image import Image
import os

from typing import List, Optional, Set 

class HoleyDataset(Dataset):

    def __init__(self, root='./data', train=True, download=True, transform=None, classes: Optional[Set[int]] = None):
        super().__init__()
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train,
                                                    download=download, transform=None) 
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.classes = classes
        if classes:
            self.indices = [i for i, (_, label) in enumerate(self.cifar10) if label in classes]
        else:
            self.indices = list(range(len(self.cifar10)))


    def __getitem__(self, original_index):
 
        index = self.indices[original_index]

        img, label = self.cifar10[index] # img is a PIL Image here

    
        img_tensor = self.base_transform(img)

        img_np = np.array(img)
        holey_img_np = create_holey_image(img_np)

        img_aug1 = self.base_transform(img_np) 
        img_aug2 = self.base_transform(holey_img_np) 

        return img_aug1, img_aug2, label

    def __len__(self):
        return len(self.indices)


def get_dataloaders(batch_size: int, classes: Optional[Set[int]] = None, **kwargs):
    """
    A simple handler to retrieve CIFAR10 contrastive loaders.
    """

    trainset = HoleyDataset(train=True, classes=classes)
    testset = HoleyDataset(train=False, classes=classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader


def create_holey_image(img, max_holes: int = 200):

    if isinstance(img, Image):
        img_np = np.array(img)
    elif isinstance(img, np.ndarray):
         img_np = img
    else:
        print(f"Warning: create_holey_image received unexpected input type: {type(img)}")
        try:
            img_np = np.array(img) 
        except Exception as e:
            print(f"Could not convert input to NumPy array: {e}")
            return img 

    # copy to avoid modifying the original image in place
    holey_img = img_np.copy()

    # Color or Grayscale image
    if holey_img.ndim == 3:
        height, width, channels = holey_img.shape
    elif holey_img.ndim == 2:
         height, width = holey_img.shape
         channels = 1 # Grayscale
    else:
         print(f"Warning: Unexpected image dimensions: {holey_img.shape}")
         return img_np 

    num_holes = random.randint(1, max_holes)

    for _ in range(num_holes):
        rand_i = random.randint(0, height - 1)
        rand_j = random.randint(0, width - 1) 

        if 0 <= rand_i < height and 0 <= rand_j < width:
             if holey_img.ndim == 3:
                holey_img[rand_i, rand_j, :] = 0 
             elif holey_img.ndim == 2:
                 holey_img[rand_i, rand_j] = 0 


    return holey_img


def visualize_random_batch(loader, directory, model=None):
    """
    Visualizes a random batch of images from the HoleyDataset loader,
    Shows the original augmented image, the holey image, and optionally a model's prediction.
    """
    # Get one batch from the loader
    # The batch contains (img_aug1, img_aug2, label)
    try:
        batch_img1, batch_img2, batch_labels = next(iter(loader))
    except StopIteration:
        print("DataLoader is empty.")
        return
    
    device = next(model.parameters()).device 
    batch_img1 = batch_img1.to(device)

    img1_tensor = batch_img1[0] # Shape (C, H, W)
    img2_tensor = batch_img2[0] # Shape (C, H, W)
    label = batch_labels[0] # Scalar label

    # Denormalize: img * std + mean
    # For CIFAR10 "normalization" (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    # put tensors on cpu
    img1_denorm = img1_tensor.cpu() * std + mean
    img2_denorm = img2_tensor.cpu() * std + mean

    # (C, H, W) tensor to (H, W, C) NumPy array and clip values to [0, 1]
    img1_np = img1_denorm.permute(1, 2, 0).numpy().clip(0, 1)
    img2_np = img2_denorm.permute(1, 2, 0).numpy().clip(0, 1)

    # --- Plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(9, 4)) 

    axs[0].imshow(img1_np)
    axs[0].set_title("Original Augmented")
    axs[0].axis('off')
   
    axs[1].imshow(img2_np)
    axs[1].set_title("Holey Image")
    axs[1].axis('off') # Hide axes

    if model is not None:
        try:
            model.eval() 
            with torch.no_grad():
                prediction_batch = model(batch_img1)
            pred_img_tensor = prediction_batch[0].cpu()
            pred_denorm = pred_img_tensor * std + mean
            pred_np = pred_denorm.permute(1, 2, 0).numpy().clip(0, 1)

            axs[2].imshow(pred_np)
            axs[2].set_title("Model Prediction")
            axs[2].axis('off') 
        except Exception as e:
            print(f"Could not get or visualize model prediction: {e}")
            axs[2].set_title("Prediction Error")
            axs[2].axis('off')
    else:
         axs[2].set_title(f"Label: {label.item()}") 
         axs[2].axis('off')

    img_directory = os.path.join(directory, "img")

    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
        print(f"Folder created : {img_directory}") 
    image_filename = "results.png" 
    full_image_path = os.path.join(img_directory, image_filename)

    plt.savefig(full_image_path)
    print(f"Image saved at : {full_image_path}")
    plt.show() 

