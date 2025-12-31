import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels = input_dim, out_channels = output_dim, kernel_size = 3, padding = 1)
        # cnn window içerisindeki lokal pattern
        self.lstm = nn.LSTM(input_size = output_dim, hidden_size = hidden_dim, batch_first = True)
        # lstm, bu patternlerin zaman içindeki akışı
        self.fc = nn.Linear(in_features = hidden_dim, out_features = num_classes)

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
            print(f"x shape: {x.shape}") # x shape: torch.Size([32, 50, 1152])
            print(f"x shape after permute {x.shape}") # x shape after permute torch.Size([32, 1152, 50])
            print(f"x shape after conv1d {x.shape}") # x shape after conv1d torch.Size([32, 64, 50]) 
            print(f"x shape after lstm {out.shape}") # x shape after lstm torch.Size([32, 50, 128])
            print(f"last timestep shape {x.shape}") # last timestep shape torch.Size([32, 128])
            print(f"x shape after fc {x.shape}") # x shape after fc torch.Size([32, 6]) 
        """
        x = x.permute(0,2,1)  # convert from (batch_size, seq_len, features) <-> (batch_size, features, seq_len) for conv1d
        x = self.conv1d(x)
        x = self.batchnorm(x)  
        x = self.relu(x)
        x = x.permute(0,2,1)  # convert from (batch_size, features, seq_len) <-> (batch_size, seq_len, features) for lstm
        out, (hidden, cell) = self.lstm(x) 
        x = out[:,-1,:]
        x = self.fc(x)
        return x