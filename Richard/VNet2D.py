import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import functools
import time
from typing import Optional, Callable, Dict, Any, Union, List, Tuple
from Richard.utils import debug_decorator

# Create module-level logger
logger = logging.getLogger(__name__)

class conv2d_block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 bn=False, 
                 activation_fn:torch.nn.Module=None,
                 debug=False):
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.activation_fn = activation_fn if activation_fn is not None else nn.ReLU(inplace=False)
        
    @debug_decorator
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        return x
    
class deconv2d_block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 bn=False, 
                 activation_fn:torch.nn.Module=None,
                 debug=False):
        super().__init__()
        self.debug = debug
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.activation_fn = activation_fn if activation_fn is not None else nn.ReLU(inplace=False)
        
    @debug_decorator
    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        return x

class downsample_block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 n_convs,
                 conv_kernel_size=5,
                 conv_stride=1, 
                 conv_padding=2,
                 conv_bn=False, 
                 conv_activation_fn:torch.nn.Module=None,
                 downconv_kernel_size=2,
                 downconv_stride=2,
                 downconv_padding=0,
                 downconv_bn=False,
                 downconv_activation_fn:torch.nn.Module=None,
                 dropout_rate=0.0,
                 debug=False):
        super().__init__()
        self.debug = debug
        self.block_name = f"down_{in_channels}_{out_channels}"
        
        # Default activations if None
        _conv_activation = conv_activation_fn if conv_activation_fn is not None else nn.ReLU(inplace=False)
        _downconv_activation = downconv_activation_fn if downconv_activation_fn is not None else nn.PReLU(num_parameters=out_channels)
            
        # Save residual activation
        self.residual_activation = _conv_activation
        
        # Set up downsample and convolution blocks
        self.downsample = conv2d_block(in_channels, out_channels, downconv_kernel_size, downconv_stride, 
                                      downconv_padding, downconv_bn, _downconv_activation, debug=debug)

        # conv_block apply convolutions in series to the output of the downsample layer (n_convs times)
        self.conv_blocks = nn.ModuleList()
        for i in range(n_convs):
            self.conv_blocks.append(conv2d_block(out_channels, out_channels, conv_kernel_size, conv_stride, 
                                                conv_padding, conv_bn, _conv_activation, debug=debug))
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        # Add Dropout layer
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    @debug_decorator
    def forward(self, x):
        x = self.downsample(x)
        residual_x = x
        x = self.conv_blocks(x)
        x = x + residual_x
        x = self.residual_activation(x)
        x = self.dropout(x)
        return x

class upsample_block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 n_convs,
                 conv_kernel_size=5,
                 conv_stride=1, 
                 conv_padding=2,
                 conv_bn=False,
                 conv_activation_fn:torch.nn.Module=None,
                 upconv_kernel_size=2,
                 upconv_stride=2,
                 upconv_padding=0,
                 upconv_bn=False,
                 upconv_activation_fn:torch.nn.Module=None,
                 dropout_rate=0.0,
                 debug=False):
        super().__init__()
        self.debug = debug
        self.block_name = f"up_{in_channels}_{out_channels}"

        # Default activations if None
        _conv_activation = conv_activation_fn if conv_activation_fn is not None else nn.ReLU(inplace=False)
        _upconv_activation = upconv_activation_fn if upconv_activation_fn is not None else nn.PReLU(num_parameters=out_channels//2)

        # Save residual activation
        self.residual_activation = _conv_activation
        
        # Set up upsample and convolution blocks
        self.upsample = deconv2d_block(in_channels, out_channels//2, upconv_kernel_size, upconv_stride, 
                                      upconv_padding, upconv_bn, _upconv_activation, debug=debug)

        # conv_block apply convolutions in series to the output of the upsample layer (n_convs times)
        self.conv_blocks = nn.ModuleList()
        for i in range(n_convs):
            self.conv_blocks.append(conv2d_block(out_channels, out_channels, conv_kernel_size, conv_stride, 
                                               conv_padding, conv_bn, _conv_activation, debug=debug))
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        # Add Dropout layer
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    @debug_decorator
    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        residual_x = x
        x = self.conv_blocks(x)
        x = x + residual_x
        x = self.residual_activation(x)
        x = self.dropout(x)
        return x

class VNet2D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=2,
                 conv_kernel_size=5,
                 conv_stride=1, 
                 conv_padding=2,
                 conv_bn=False,
                 conv_activation_fn:torch.nn.Module=None,
                 sampling_conv_kernel_size=2,
                 sampling_conv_stride=2,
                 sampling_conv_padding=0,
                 sampling_conv_bn=False,
                 sampling_conv_activation_fn:torch.nn.Module=None,
                 dropout_rate=0.0,
                 debug=False):
        super().__init__()
        self.debug = debug
        self.init_conv = conv2d_block(in_channels, 16, conv_kernel_size, conv_stride, conv_padding, 
                                     conv_bn, conv_activation_fn, debug=debug)
                                     
        self.downsample_block_1 = downsample_block(16, 32, 2, conv_kernel_size, conv_stride, conv_padding, 
                                                  conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                                  sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                                  sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
                                                  
        self.downsample_block_2 = downsample_block(32, 64, 3, conv_kernel_size, conv_stride, conv_padding, 
                                                  conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                                  sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                                  sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
                                                  
        self.downsample_block_3 = downsample_block(64, 128, 3, conv_kernel_size, conv_stride, conv_padding, 
                                                  conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                                  sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                                  sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
                                                  
        self.downsample_block_4 = downsample_block(128, 256, 3, conv_kernel_size, conv_stride, conv_padding, 
                                                  conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                                  sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                                  sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
        
        self.upsample_block_1 = upsample_block(256, 256, 3, conv_kernel_size, conv_stride, conv_padding, 
                                              conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                              sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                              sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
                                              
        self.upsample_block_2 = upsample_block(256, 128, 3, conv_kernel_size, conv_stride, conv_padding, 
                                              conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                              sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                              sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
                                              
        self.upsample_block_3 = upsample_block(128, 64, 2, conv_kernel_size, conv_stride, conv_padding, 
                                              conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                              sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                              sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
                                              
        self.upsample_block_4 = upsample_block(64, 32, 1, conv_kernel_size, conv_stride, conv_padding, 
                                              conv_bn, conv_activation_fn, sampling_conv_kernel_size, 
                                              sampling_conv_stride, sampling_conv_padding, sampling_conv_bn, 
                                              sampling_conv_activation_fn, dropout_rate=dropout_rate, debug=debug)
    
        self.final_conv = conv2d_block(32, out_channels, 1, 1, 0, conv_bn, conv_activation_fn, debug=debug)
        
    @debug_decorator
    def forward(self, x):
        x_down_16 = self.init_conv(x)
        x_down_32 = self.downsample_block_1(x_down_16)
        x_down_64 = self.downsample_block_2(x_down_32)
        x_down_128 = self.downsample_block_3(x_down_64)
        x_down_256 = self.downsample_block_4(x_down_128)
        
        x_up_256 = self.upsample_block_1(x_down_256, x_down_128)
        x_up_128 = self.upsample_block_2(x_up_256, x_down_64)
        x_up_64 = self.upsample_block_3(x_up_128, x_down_32)
        x_up_32 = self.upsample_block_4(x_up_64, x_down_16)
        
        x = self.final_conv(x_up_32)
        
        return x
                
    @torch.no_grad()
    def check_gradient_flow(self):
        """Print the mean gradient for each layer - call after backward()"""
        logger.debug("Gradient flow:")
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                logger.debug(f"{name}: mean grad {param.grad.abs().mean().item():.5f}, shape {param.shape}")
            elif param.requires_grad:
                logger.debug(f"{name}: NO GRADIENT, shape {param.shape}")
        
