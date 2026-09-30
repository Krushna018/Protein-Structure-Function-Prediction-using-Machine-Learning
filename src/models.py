import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_regression():
    return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

def get_random_forest():
    return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def get_mlp(input_dim, num_classes=6):
    return SimpleMLP(input_dim, num_classes)
