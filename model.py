import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CBAM3D

def passthrough(x, **kwargs):
    return x

# ELU integration
def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# Batch Normalization
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

# ELU Convoluation
class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)

# Down transition for VNet
class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)
        #self.cbam = CBAM3D(gate_channels=16)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)

        # Apply the CBAM attention module 
        #out = self.cbam(out)

        out = self.relu2(torch.add(out, down))
        
        return out

# Up Transition for VNet
class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

# Output Transition for VNet
class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)

        # Somewhere in here: CAM Attention mechanism & BAM Attention machanism
        
        if nll:
            self.softmax = lambda x: F.log_softmax(x, dim=1)
        else:
           self.softmax = lambda x: F.softmax(x, dim=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        #out = out.permute(0, 2, 3, 4, 1).contiguous()

        # flatten
        #out = out.view(out.numel() // 2, 2)
        #out = self.softmax(out.permute(0,4,1,2,3))

        #out = out.permute(0,4,1,2,3)
        # treat channel 0 as the predicted output
        return out

# Define the separation operation for DigSep module
def sep_operation(x, thresh):
    
    # Get digital feature map
    # If tensor is greater than or equal to the threshold, set equal to the threshold. Otherwise, set equal to 0
    x2 = torch.where(x >= thresh, thresh, 0)

    # Apply digital feature map to the original image
    x = x - x2

    return x

# Defining DigSep module as described in improved VNet paper
class DigSep(nn.Module):
    def __init__(self, inChans, elu):
        super(DigSep, self).__init__()

        # May need to change settings here
        self.conv1 = nn.Conv3d(16, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)
        
    # Input x here needs to be one slice?
    def forward(self, x):

        # Perform the thresholding
        out128 = sep_operation(x, 128)
        out64 = sep_operation(x, 64)
        out32 = sep_operation(x, 32)
        out16 = sep_operation(x, 16)

        # This will create 16 channels from the input tensor
        x16 = x.repeat(1, 16, 1, 1, 1)  # repeat along the channel dimension
        
        # Return the 16 channels: 4*2 of the digital map images, 8 original images
        out = torch.concat((out128, out128, out64, out64, out32, out32, out16, out16, x, x, x, x, x, x, x, x), dim=1)
        # Convolution & batch normalization
        out = self.bn1(self.conv1(out))

        # Add operation
        out = self.relu1(torch.add(out, x16))

        return out

class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)
        #self.cbam = CBAM3D(gate_channels=16)

    def forward(self, x):
        # Apply batch norm and convolution
        out = self.bn1(self.conv1(x))

        # Expand input tensor along channel dimension (dim=1)
        # This will create 16 channels from the input tensor
        #x16 = x.repeat(1, 16, 1, 1, 1)  # repeat along the channel dimension

        # Apply the CBAM attention module 
        #out = self.cbam(out)

        # Add the expanded input tensor to the output of the convolution
        #out = self.relu1(torch.add(out, x16))

        return out

class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        #self.digsep = DigSep(16, elu)
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):
        #out16 = self.digsep(x)
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
