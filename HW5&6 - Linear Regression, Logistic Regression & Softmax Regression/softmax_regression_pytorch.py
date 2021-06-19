import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

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

if __name__ == '__main__':
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    import torch

    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    # Load data
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    print(X_train.shape, y_train.shape)

    X_train_ts = torch.Tensor(normalize(X_train)).to(device)
    y_train_ts = torch.LongTensor(y_train).to(device)
    datasets = TensorDataset(X_train_ts, y_train_ts)
    train_iter = DataLoader(datasets, batch_size=1, shuffle=True)

    model = SoftmaxRegression(X_train_ts.shape[1], 3)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trained_model = train(train_iter, model, criterion, optimizer, epochs=1000)


