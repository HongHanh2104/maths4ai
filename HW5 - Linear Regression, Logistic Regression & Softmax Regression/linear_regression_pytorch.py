import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self, w_num):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(w_num, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x)

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