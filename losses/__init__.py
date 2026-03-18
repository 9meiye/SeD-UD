from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'MSELoss': nn.MSELoss(),
            }