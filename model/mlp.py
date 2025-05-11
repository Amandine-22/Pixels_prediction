import torch.nn as nn

class MLP(nn.Module):
    """
    A small MLP adapted to become a sort of autoencoder

    Args: 
        input_dim
        output_dim
        hidden_dim
        n_layers
    """
    # Réflexe à avoir dans le constructeur, passer un **kwargs qui permet d'éviter les erreurs dues au passage d'arguments invalides 
    def __init__(self, input_dim=3072, output_dim=3072, hidden_dim=8000, n_layers=2, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers = []

        # Couche d'entrée
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU())

        # car la première couche (input_dim -> hidden_dim) a déjà été ajoutée.
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())

        # Couche de sortie
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # On utilise *layers pour décompresser la liste.
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        B, C, W, H = x.shape
        x = x.reshape(B, C*H*W)
        x = self.layers(x)
        x = x.view(B, C, H, W)
        return x