import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from sklearn.metrics import f1_score

def train_mlp(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=1e-3, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate class weights for imbalance
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_X = torch.FloatTensor(X_val).to(device)
    val_y = torch.LongTensor(y_val).to(device)
    
    best_val_f1 = -1
    best_model = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(val_X)
            _, preds = torch.max(outputs, 1)
            val_f1 = f1_score(val_y.cpu(), preds.cpu(), average='macro')
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    model.load_state_dict(best_model)
    return model

def predict_mlp(model, X_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_X = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(test_X)
        _, preds = torch.max(outputs, 1)
        
    return preds.cpu().numpy()
