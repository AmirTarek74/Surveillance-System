import torch
import torch.nn as nn 
import torchvision

class ActionRecognition(nn.Module):
    def __init__(self,num_classes):
        
        super(ActionRecognition, self).__init__()
        self.num_classes =num_classes
        self.weigths =  torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.efficient = torchvision.models.efficientnet_b0(weights=self.weigths)

        self.conv_layers = self.efficient.features
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        self.lstm_layers = nn.LSTM(input_size=1280, hidden_size=128, num_layers=2, batch_first=True)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x):
        batch_size, num_frames, channels, height, width = x.size()
        
        x = x.view(batch_size * num_frames, channels, height, width)
        
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(batch_size, num_frames, -1)
        
        _, (h_n, _) = self.lstm_layers(x)
        
        x = h_n[-1, :, :]
        
        x = self.fc_layers(x)
        return x