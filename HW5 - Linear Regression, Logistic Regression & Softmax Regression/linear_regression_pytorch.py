import torch
import torch.nn as nn
#from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def train(data, model, criterion, optimizer, epochs):
    X, y = data
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, loss {loss.item()}')
    print('Complete training')
    return model

def predict(data, model):
    X = data
    with torch.no_grad():
        out = model(X).data.numpy()
    return out

def test():
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    X_train = torch.randn(15, 1).to(device)
    y_train = torch.randn(15, 1).to(device)
    
    #X_train = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
    #y_train = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
    
    model = LinearRegression()
    model = model.to(device)

    criterion = torch.nn.MSELoss(size_average = False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    trained_model = train((X_train, y_train), model, criterion, optimizer, 500)
    result = predict(X_train, trained_model)
    print(result)

if __name__ == '__main__':
    test()