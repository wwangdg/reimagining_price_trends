import torch
import torch.nn as nn
from torchsummary import summary

class ConvNet(nn.Module):
    """
    Dynamically constructs the core building blocks (convolution, activation, and pooling) based on the 'layers' parameter.
    Automatically calculates the input dimensions for the fully connected layer based on the information from the core building blocks.
    Employs techniques such as Xavier initialization, dropout, and batch normalization to optimize the model's performance.
    Suitable for both classification and regression problems
    """
    def __init__(self, layers, input_channels=1, out_dim=2, dropout=0.5, BN=True, xavier=True, activation_fn=nn.LeakyReLU, H_in=64, W_in=60):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential()
        self.input_channels = input_channels
        self.activation_fn = activation_fn
        self.H_in = H_in
        self.W_in = W_in
        
        for i, layer_config in enumerate(layers):
            out_channels, kernel_size, stride, dilation, max_pool = layer_config
            
            if i == 0:
                padding0 = ((self.H_in-1)*stride[0]-H_in+dilation[0]*(kernel_size[0]-1)+1)//2
                padding1 = ((self.W_in-1)*stride[1]-W_in+dilation[1]*(kernel_size[1]-1)+1)//2
                padding=(padding0,padding1)
            else:
                padding='same'
            conv = nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,padding=padding)
            if xavier:
                nn.init.xavier_uniform_(conv.weight)
            
            if BN:
                bn = nn.BatchNorm2d(out_channels)
            
            act = self.activation_fn()
            
            pool = nn.MaxPool2d(max_pool)
            
            self.features.add_module(f'conv{i+1}', conv)
            if BN:
                self.features.add_module(f'bn{i+1}', bn)
            self.features.add_module(f'act{i+1}', act)
            self.features.add_module(f'pool{i+1}', pool)
            
            input_channels = out_channels
        
        input_shape = (self.input_channels, self.H_in, self.W_in)
        self.flattened_size = self._get_conv_output(input_shape)
        self.fc = nn.Linear(self.flattened_size, out_dim)
        if xavier:
            nn.init.xavier_uniform_(self.fc.weight)
        
        self.dropout = nn.Dropout(p=dropout)
        
        print(self)
        summary(self, input_size=input_shape, device="cpu")
        
        
    def forward(self, x):
        out = self.features(x)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        result = self.fc(out)
        return result
    
    def _get_conv_output(self, input_shape):
        with torch.no_grad():
            batch_size = 1
            input = torch.rand(batch_size, *input_shape)
            output = self.features(input)
            return int(torch.numel(output) / batch_size)