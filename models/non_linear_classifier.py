import torch.nn as nn

# MLP(20)
class NonLinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes, num_hidden=20, layer_norm=False):
        super(NonLinearClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm
        
        self.hidden1 = nn.Linear(self.in_features, self.num_hidden)  
        self.activation = nn.ReLU()                    
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(self.num_hidden)
        self.hidden2 = nn.Linear(self.num_hidden, self.num_classes) 

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        if self.layer_norm:
            x = self.layer_norm1(x)
        x = self.hidden2(x)
        return x


# MLP(128)
class TwoLayerClassifier(nn.Module):
    def __init__(self, in_features, num_classes, num_hidden, layer_norm=False):
        super(TwoLayerClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm
        
        self.hidden1 = nn.Linear(self.in_features, self.num_hidden)  
        self.activation = nn.ReLU()                 
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(self.num_hidden)
        self.hidden2 = nn.Linear(self.num_hidden, self.num_classes)  

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        if self.layer_norm:
            x = self.layer_norm1(x)
        x = self.hidden2(x)
        return x
