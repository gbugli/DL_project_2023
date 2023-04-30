import torch.nn as nn
import torch
from ConvLSTM import ConvLSTM

class Seq2Seq(nn.Module):

    def __init__(self, num_channels, pred_frames, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers, device='mps'):

        super().__init__()

        self.device = device
        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                pred_frames = pred_frames, kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size, device=self.device)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    pred_frames = pred_frames, kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size, device=self.device)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):

        # Forward propagation through all the layers
        print('Shape before seq')
        print(X.shape)
        output = self.sequential(X)
        print('Shape after sequence')
        print(output.shape)

        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        
        return nn.Sigmoid()(output)

    