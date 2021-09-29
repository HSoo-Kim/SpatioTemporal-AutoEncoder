import torch
import torch.nn as nn
from Models.ConvLSTM import ConvLSTM

class ST_AutoEncoder(nn.Module):
    """
    Sequential Model for the Spatio Temporal Autoencoder (ST_AutoEncoder)
    """
    
    def __init__(self, in_channel):
        super(ST_AutoEncoder, self).__init__()
        
        self.in_channel = in_channel
        
        # Spatial Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channel, out_channels=128, kernel_size=(1,11,11), stride=(1,4,4)),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1,5,5), stride=(1,2,2)),
            nn.ReLU()            
        )
                
        # Temporal Encoder & Decoder
        self.temporal_encoder_decoder = Temporal_EncDec()
        
        # Spatial Decoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=128, kernel_size=(1,5,5), stride=(1,2,2)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=3, kernel_size=(1,11,11), stride=(1,4,4)),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x : (N, C, T, H, W) / T : sequence length(or D)        
        h = self.spatial_encoder(x)
        
        h = h.permute(0,2,1,3,4)  # (N, C, T, H, W) -> (N, T, C, H, W) 
        h_hat = self.temporal_encoder_decoder(h)
        
        h_hat = h_hat.permute(0,2,1,3,4)  # (N, T, C, H, W) -> (N, C, T, H, W) 
        output = self.spatial_decoder(h_hat)
        return output
    
    
class Temporal_EncDec(nn.Module):
    def __init__(self):
        super(Temporal_EncDec, self).__init__()
        
        self.convlstm_1 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_2 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_3 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3,3), num_layers=1, bias=True)
        
    def forward(self, x):
        layer_output_list, _ = self.convlstm_1(x)
        layer_output_list, _ = self.convlstm_2(layer_output_list[0])
        layer_output_list, _ = self.convlstm_3(layer_output_list[0])
        
        return layer_output_list[0]
