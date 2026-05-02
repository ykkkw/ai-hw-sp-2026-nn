import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, 
                 input_size = 28*28, 
                 hidden_size = [256, 128], 
                 output_size = 10,
                 dropout = 0.3
                 ):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.layer2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.layer3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)

        out = self.layer1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.layer2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.layer3(out)
        return out
    
class CNN(nn.Module):
    def __init__(self, 
                 inchannel = 1,
                 conv_channel = [32, 64, 128],
                 kernel_size = 3,
                 conv_dropout = 0.25,
                 fc_size = 256,
                 fc_dropout = 0.5,
                 output_size = 10,
                 ):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, conv_channel[0], kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(conv_channel[0], conv_channel[1], kernel_size)
        self.relu2 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(conv_dropout)

        self.conv3 = nn.Conv2d(conv_channel[1], conv_channel[2], kernel_size)
        self.relu3 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(conv_dropout)

        fc_input = conv_channel[2] * 5 * 5
        self.fc1  = nn.Linear(fc_input, fc_size)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(fc_dropout)
        self.fc2  = nn.Linear(fc_size, output_size)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pooling1(out)
        out = self.dropout1(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pooling2(out)
        out = self.dropout2(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out
    
class Embedding(nn.Module):
    def __init__(self, 
                patch_height=4, 
                img_width=28, 
                d_model=128):   
        super(Embedding, self).__init__()
        self.patch_height = patch_height
        self.proj = nn.Linear(patch_height * img_width, d_model)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.unfold(1, self.patch_height, self.patch_height)
        x = x.reshape(x.size(0), x.size(1), -1)
        return self.proj(x)

class Encoder(nn.Module):
    def __init__(self, 
                 d_model=128, 
                 nhead=4, 
                 num_layers = 3,
                 forward_dim = 256,
                 dropout=0.1,
                 output_size = 10,
                 patch_height = 4,
                 img_width = 28
                 ):
        super(Encoder, self).__init__()
        self.embedding = Embedding(patch_height, img_width, d_model)
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=forward_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, output_size)

    def forward(self, x):
        B = x.size(0)
        out = self.embedding(x)
        cls = self.token.expand(B, -1, -1)
        out = torch.cat([cls, out], dim=1)
        out = self.transformer_encoder(out)
        return self.head(out[:, 0])