import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell

device = torch.device('mps')

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, pred_frames,
    kernel_size, padding, activation, frame_size, device):

        super().__init__()
        self.pred_frames = pred_frames
        self.device = device
        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size, device=device)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, ch, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len + self.pred_frames, 
        height, width, device=self.device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=self.device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=self.device)

        X = torch.cat([X, torch.zeros((batch_size, ch, self.pred_frames, height, width)).to(self.device)], dim=2)

        # Unroll over time steps
        for time_step in range(seq_len+self.pred_frames):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output

        