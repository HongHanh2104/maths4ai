import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRegression(nn.Module):
    def __init__(self, w_num, C):
        super(SoftmaxRegression, self).__init__()
        self.softmax = nn.Linear(w_num, C, bias=True)
    
    def forward(self, x):
        return F.softmax(self.softmax(x))

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

