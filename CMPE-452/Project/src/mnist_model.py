import torch.nn as nn
import torch
from torch.autograd import Function
import math

class Euclid_Squared(Function):
    '''torch.autograd.Function child class with static forward and backward methods for euclidnet'''
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input,weights)

        # Prepare dims for euclidean distance calculation
        input_b = input.unsqueeze(1) # batch x 1 x inputlayer
        weights_b = weights.mT.unsqueeze(0) # 1 x outputlayer x inputlayer
        weights_b = weights_b.broadcast_to((input.shape[0], weights_b.shape[1], weights_b.shape[2])) # batch x outputlayer x inputlayer

        # Output layer calculation
        output = (input_b-weights_b)**2 # batch x 1 x outputlayer
        output = output.sum(2) # batch x outputlayer

        return (-1/2)*output
    
    @staticmethod
    def backward(ctx,grad_output):
        input,weights = ctx.saved_tensors

        # Prepare dims for euclidean distance calculation
        input_b = input.unsqueeze(2) 
        weights_b = weights.unsqueeze(0) 
        weights_b = weights_b.broadcast_to((input.shape[0], weights_b.shape[1], weights_b.shape[2])) 
        grad_output_b = grad_output.unsqueeze(1)

        # Calculate gradient for weights
        grad_weights = (weights_b-input_b)*grad_output_b

        # Calculate gradient for input
        grad_input = (input_b-weights_b)*grad_output_b
        grad_input_s = grad_input.sum(2)

        #test = 1/0 #- For debugging backward pass, doesn't like vscode breakpoints, maybe because it's calling C++ engine eventually?
        
        return -grad_input_s, -grad_weights


class Euclid_FC(nn.Module):
    '''torch.nn.Module child class to build fully connnected euclid layer'''
    def __init__(self, input, output):
        super(Euclid_FC, self).__init__()
        self.input_size = input
        self.output_size = output
        self.weights = nn.Parameter(nn.init.normal_(torch.randn((self.input_size, self.output_size)), mean=0, std=1))

    def forward(self, x):
        output = Euclid_Squared.apply(x, self.weights)
        return output


class mnist_classifier(nn.Module):
    def __init__(self, N_input=784, N_output=10):
        super(mnist_classifier, self).__init__()
        N2 = 392
        N3 = 128
        self.fc1 = Euclid_FC(N_input, N2)
        self.bn1 = nn.BatchNorm1d(N2)
        self.fc2 = Euclid_FC(N2, N3)
        self.bn2 = nn.BatchNorm1d(N3)
        self.fc3 = Euclid_FC(N3, N_output)
        self.bn3 = nn.BatchNorm1d(N_output)
        self.type = 'MLP4'
        self.input_shape = (1,784)

    def forward(self, X):
        X = self.fc1(X)
        X = self.bn1(X)
        X = nn.functional.relu(X)
        X = self.fc2(X)
        X = self.bn2(X)
        X = nn.functional.relu(X)
        X = self.fc3(X)
        X = self.bn3(X)
        return X


class mnistautoencoder(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(mnistautoencoder, self).__init__()
        N2 = 392
        self.fc1 = Euclid_FC(N_input, N2)
        self.bn1 = nn.BatchNorm1d(N2)
        self.fc2 = Euclid_FC(N2, N_bottleneck)
        self.bn2 = nn.BatchNorm1d(N_bottleneck)
        self.fc3 = Euclid_FC(N_bottleneck, N2)
        self.bn3 = nn.BatchNorm1d(N2)
        self.fc4 = Euclid_FC(N2, N_output)
        self.bn4 = nn.BatchNorm1d(N_output)
        self.type = 'MLP4'
        self.input_shape = (1,784)

    def forward(self, X):
        #encoder
        X = self.fc1(X)
        X = self.bn1(X)
        X = nn.functional.relu(X)
        X = self.fc2(X)
        X = self.bn2(X)
        X = nn.functional.relu(X)

        #decoder
        X = self.fc3(X)
        X = self.bn3(X)
        X = nn.functional.relu(X)
        X = self.fc4(X)
        X = self.bn4(X)
        X = nn.functional.sigmoid(X)

        return X
    
    def encode(self, X):
        X = self.fc1(X)
        X = self.bn1(X)
        X = nn.functional.relu(X)
        X = self.fc2(X)
        X = self.bn2(X)
        X = nn.functional.relu(X)
        return X
    
    def decode(self, X):
        X = self.fc3(X)
        X = self.bn3(X)
        X = nn.functional.relu(X)
        X = self.fc4(X)
        X = self.bn4(X)
        X = nn.functional.sigmoid(X)
        return X