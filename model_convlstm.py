import torch
import torch.nn as nn

class ConvLSTM_Model(nn.Module):
    def __init__(self, in_channels=2, num_class=12, sequence_length=30, num_keypoints=17):
        super(ConvLSTM_Model, self).__init__()
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints

        # 1D Convolutional layers with residual connections
        self.conv1 = nn.Conv1d(in_channels * num_keypoints, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.3)

        # Residual adapters
        self.res_adapter1 = nn.Conv1d(in_channels * num_keypoints, 64, kernel_size=1) if in_channels * num_keypoints != 64 else nn.Identity()
        self.res_adapter2 = nn.Conv1d(64, 128, kernel_size=1) if 64 != 128 else nn.Identity()

        # LSTM input size
        lstm_in_size = 128  # Output channels from conv4

        # LSTM layer
        self.lstm = nn.LSTM(lstm_in_size, 256, num_layers=2, batch_first=True, dropout=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Input x: [B, T, V*C] -> Reshape to [B, V*C, T]
        B, T, VC = x.shape
        x = x.view(B, T, self.num_keypoints, -1)  # [B, T, V, C]
        x = x.permute(0, 3, 2, 1).reshape(B, VC, T)  # [B, V*C, T]

        # Convolutional layers with residual connections
        residual = self.res_adapter1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + residual  # Residual connection
        x = self.pool(x)  # [B, 64, T//2]
        x = self.dropout_conv(x)

        residual = self.res_adapter2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + residual  # Residual connection
        x = self.pool(x)  # [B, 128, T//4]
        x = self.dropout_conv(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # [B, T//4, 128]

        # LSTM layer
        lstm_out, _ = self.lstm(x)  # [B, T//4, 256]
        x = lstm_out[:, -1, :]  # Last time step [B, 256]

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x