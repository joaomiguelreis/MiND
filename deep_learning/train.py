import torch
import pandas as pd
from model.CNN import TwoLayerNet


# N is size of the training set; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 30, 5, 10, 2

# Data
PT_data = pd.read_excel("../PTResults_trimmed.xlsx")
PT_tensor = torch.tensor(PT_data.values)

#Eventually change to batch training and shuffle what is training
x_train = PT_tensor[:N,1:-1].float()
y_train = PT_tensor[:N,-1].long()#.view(N,1)

x_test = PT_tensor[N:,1:-1].float()
y_test = PT_tensor[N:,-1].long()#.view(N,1)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    print(t, loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Test accuracy 
    correct_train = (torch.argmax(y_pred, dim=-1)==y_train).sum()
    acc_train = correct_train.float()/y_train.shape[0]

    y_hat = model(x_test)
    correct_test = (torch.argmax(y_hat, dim=-1)==y_test).sum()
    acc_test = correct_test.float()/y_test.shape[0]

    print('Training accuracy: ', acc_train.item())
    print('Testing accuracy: ', acc_test.item())


