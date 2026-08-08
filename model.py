import torch.nn as nn
# Define Artificial Neural Network
class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Input: 784 → Hidden Layer: 256
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Hidden Layer: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Output Layer: 128 → 10
            nn.Linear(128, 10)
        )

    def forward(self, x):

        # Flatten [batch, 1, 28, 28]
        # into [batch, 784]
        x = x.view(x.size(0), -1)

        # Pass flattened data through ANN
        return self.network(x)