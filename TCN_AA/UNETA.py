import torch
import torch.nn as nn
from torchvision.transforms import Resize
from TCN import TemporalConvNet
from TCN import BidirectionalTCN

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class encoder_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv = conv_block(in_channel, out_channel)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.Wg = nn.Sequential(
            nn.Conv2d(in_channel[0], out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
        
        self.Ws = nn.Sequential(
            nn.Conv2d(in_channel[1], out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0),
                                    nn.Sigmoid())
        
    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s
    
class decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, biliner=True):
        super().__init__()
        
        if biliner:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel[0], in_channel[0], kernel_size=2, stride=2)
        self.ag = attention_gate(in_channel, out_channel)
        self.c1 = conv_block(in_channel[0] + out_channel, out_channel)
        
    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x
    
class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        c = [2 ** _ for _ in range(5, 13)]
        self.e1 = encoder_block(9, c[0])
        self.e2 = encoder_block(c[0], c[1])
        self.e3 = encoder_block(c[1], c[2])
        self.e4 = encoder_block(c[2], c[3])
        
        self.b1 = conv_block(c[3], c[4])
        
        self.d1 = decoder_block([c[4], c[3]], c[3])
        self.d2 = decoder_block([c[3], c[2]], c[2])
        self.d3 = decoder_block([c[2], c[1]], c[1])
        self.d4 = decoder_block([c[1], c[0]], c[0])
        
        # Output is number of keypoint
        self.output = nn.Conv2d(c[0], 17, kernel_size=1, padding=0)
        
        self.linear = nn.Linear(64, 2)
        
        self.avg = nn.AdaptiveAvgPool1d(2)
        
        # input sequence is number of subcarrier
        self.tcn = TemporalConvNet(30, [30, 50, 75], 3, 0.5)
        # This model use Bidirectional TCN like Bidirection LSTM
        self.bitcn = BidirectionalTCN(30, [30, 50, 75], 3, 0.5)
        
    def forward(self, x):
       
        #tcn
        '''
        tcn_out = self.tcn(x[:,0,:,:]).unsqueeze(1)
        for i in range(1,9):
            out = self.tcn(x[:,i,:,:]).unsqueeze(1)
            tcn_out = torch.cat([tcn_out,out],1)
        x = tcn_out
        '''
        #bitcn
        bitcn_out = self.bitcn(x[:, 0, :, :]).unsqueeze(1)
        for i in range(1, 9):
            out = self.bitcn(x[:, i, :, :]).unsqueeze(1)
            bitcn_out = torch.cat([bitcn_out, out], 1)
        x = bitcn_out
        
        resizer = Resize((64, 64), antialias=None) # batch, 9, 64, 64
        x = resizer(x)
        
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        b1 = self.b1(p4)
        
        d1 = self.d1(b1, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        output = self.output(d4)
        output = self.linear(output[:,:,-1])

        return output