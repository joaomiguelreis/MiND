import torch
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in=5, H=10, D_out=2):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 10)
        self.linear3 = torch.nn.Linear(10, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h_relu).sigmoid()
        y_pred = self.linear3(h2_relu)
        return y_pred

class TwoLayerNetMult(torch.nn.Module):
    def __init__(self, D_in=6, H=10, D_out=2):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNetMult, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 10)
        self.linear3 = torch.nn.Linear(10, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h_relu).sigmoid()
        y_pred = self.linear3(h2_relu)
        return y_pred