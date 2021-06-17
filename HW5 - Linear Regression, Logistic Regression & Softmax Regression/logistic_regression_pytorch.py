import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, w_num):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(w_num, 1, bias=True)
    
    def forward(self, x):
        return F.sigmoid(self.logistic(x))

def train(train_iter, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        total_loss = []
        for X, y in train_iter:
            out = model(X)
            loss = criterion(out, y)
            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, loss {sum(total_loss) / len(total_loss)}')
    print('Complete training')
    return model

def predict(data, model):
    X = data
    with torch.no_grad():
        out = model(X).data.numpy()
    return out

