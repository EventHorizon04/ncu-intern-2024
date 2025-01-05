import torch
import torch.nn as nn
from TCN import BidirectionalTCN

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_c[0], in_c[0], kernel_size=2, stride=2)
        self.ag = AttentionGate(in_c, out_c)
        self.c1 = ConvBlock(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x


class DynamicAttentionUNET(nn.Module):
    def __init__(self, num_input:int, input_shape:tuple, target_size:tuple=(64,64)):
        """
        Initializes the Dynamic Attention U-Net model.

        Parameters:
            num_input (int): The number of input tensors to process.
            input_shape (tuple): A tuple containing input shape information. Example: (4,?,6).
            target_size (tuple): A tuple specifying the target size for resizing. Example: (64,64).
        """
        super().__init__()
        self.num_input:int = num_input
        self.input_shape:tuple = input_shape
        r_s = ((input_shape[0]<<2).bit_length()-1) + (num_input.bit_length()-1)
        c = [2 ** _ for _ in range(r_s, r_s+5)]
        
        # Encoder
        self.e1 = EncoderBlock(input_shape[0]*num_input, c[0]) # Example: 2 input * (4,?,?) = 18
        self.e2 = EncoderBlock(c[0], c[1])
        self.e3 = EncoderBlock(c[1], c[2])
        self.e4 = EncoderBlock(c[2], c[3])

        # Bridge
        self.b1 = ConvBlock(c[3], c[4])

        # Decoder
        self.d1 = DecoderBlock([c[4], c[3]], c[3])
        self.d2 = DecoderBlock([c[3], c[2]], c[2])
        self.d3 = DecoderBlock([c[2], c[1]], c[1])
        self.d4 = DecoderBlock([c[1], c[0]], c[0])

        self.output = nn.Conv2d(c[0], 17, kernel_size=1, padding=0) # Predict 17 keypoint maps

        self.adaptive_pool = nn.AdaptiveAvgPool2d(target_size) # Resize input for compatibility
        
        b_in = input_shape[1]
        # Initialize a list of BidirectionalTCN instances based on num_input
        self.bitcns = nn.ModuleList([BidirectionalTCN(b_in, [b_in, b_in+20, b_in+50], 3, 0.5) for _ in range(num_input)])


    def forward(self, *args):
        # Check if any arguments were provided
        if not args:
            raise ValueError("At least one input tensor is required.")
        # Check number of arguments provided match number of input
        elif len(args) != self.num_input:
            raise ValueError(f"Expected {self.num_input} input tensors, got {len(args)}")
        # Check shape of arguments provided match shape of input
        for idx, arg in enumerate(args):
            if arg.shape[1:] != self.input_shape:
                raise ValueError(f"Input shape of argument{idx+1}: {arg.shape} does not match the expected input shape {self.input_shape}")

        # Main list to hold features from each input tensor
        main_txrx_feat = []
        
        for idx, input in enumerate(args):
            txrx_feat = [] # Temporary list for the current input tensor

            # Apply the corresponding BidirectionalTCN to each txrx pair
            num_txrx = input.shape[1] # Get number of txrx pairs
            for i in range(num_txrx):
                out = self.bitcns[idx](input[:,i,:,:]) # Use the appropriate BidirectionalTCN idx
                txrx_feat.append(out.unsqueeze(1)) # append Shape: (b,1,s,p)

            # Concatenate fewatures for the current input tensor
            input_processed = torch.cat(txrx_feat, dim=1) # Shape: (b,num_txrx,s,p)
            input_processed = self.adaptive_pool(input_processed) # Shape: (b, num_txrx, target_size, target_size)

            main_txrx_feat.append(input_processed) # Store processed features

        # Concatenate all processed features from the main list
        x = torch.cat(main_txrx_feat, dim=1) # Shape: (b, num_txrx * num_input, target_size, target_size)

        # Encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bridge
        b1 = self.b1(p4)

        # Decoder
        d1 = self.d1(b1, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        output = self.output(d4)
        hm_chk = [s1,s2,s3,s4,b1,d1,d2,d3,d4]
        return output, hm_chk