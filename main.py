from model import *
from utils import get_dataloaders, train, evaluate
from torch.nn import MSELoss
from torch.optim import SGD
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = MLP().to(device)
model = RDUNet(base_filters=128, channels=3).to(device) 
optimizer = SGD(model.parameters(), lr=0.001) 
criterion = MSELoss() 

train_loader, test_loader = get_dataloaders(batch_size=100)

train(model, train_loader, optimizer, criterion, n_epochs=1, device=device)

eval_loss = evaluate(model, test_loader, criterion, device, directory="/home/emmanuel/Documents/GitHub/Pixels_prediction")
print(f"Final Evaluation Loss: {eval_loss:.4f}")