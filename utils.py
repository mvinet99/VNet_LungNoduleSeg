import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
                  
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return 1 - dice

class BasicConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate3D, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool3d(x, 2, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_3d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

def logsumexp_3d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool3D(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate3D(nn.Module):
    def __init__(self):
        super(SpatialGate3D, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class CBAM3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelGate3D(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate3D()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class DiameterLoss(nn.Module):
    def __init__(self):
        super(DiameterLoss, self).__init__()

    def compute_weighted_centroid(self, mask):
        """ Compute soft centroid as a weighted sum of coordinates. """
        B, H, W, D = mask.shape  # (Batch, Height, Width, Depth)
        device = mask.device

        # Generate coordinate grids
        y_coords = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, W, D)
        x_coords = torch.arange(W, device=device).view(1, 1, W, 1).expand(B, H, W, D)

        # Convert to same dtype as mask
        y_coords = y_coords.to(mask.dtype)
        x_coords = x_coords.to(mask.dtype)

        # Compute weighted centroid
        total_mass = mask.sum(dim=(1, 2, 3), keepdim=True) + 1e-6  # Avoid div by zero
        centroid_x = (x_coords * mask).sum(dim=(1, 2, 3), keepdim=True) / total_mass
        centroid_y = (y_coords * mask).sum(dim=(1, 2, 3), keepdim=True) / total_mass

        return centroid_x.view(B), centroid_y.view(B)  # Ensure correct shape

    def compute_diameter(self, mask):
        """ Compute a differentiable approximation of the longest diameter. """
        B, H, W, D = mask.shape  # (Batch, Height, Width, Depth)
        device = mask.device

        # Generate coordinate grids
        y_coords = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, W, D)
        x_coords = torch.arange(W, device=device).view(1, 1, W, 1).expand(B, H, W, D)

        # Convert to same dtype as mask
        y_coords = y_coords.to(mask.dtype)
        x_coords = x_coords.to(mask.dtype)

        # Compute centroid
        centroid_x, centroid_y = self.compute_weighted_centroid(mask)

        # Reshape centroid for broadcasting
        centroid_x = centroid_x.view(B, 1, 1, 1)  # Shape (B,1,1,1)
        centroid_y = centroid_y.view(B, 1, 1, 1)  # Shape (B,1,1,1)

        # Compute distances from centroid
        dist_x = (x_coords - centroid_x).abs()
        dist_y = (y_coords - centroid_y).abs()

        # Approximate max distance in x and y directions (soft max)
        max_x = torch.sum(dist_x * mask, dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
        max_y = torch.sum(dist_y * mask, dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)

        # Approximate max diameter using Pythagorean theorem
        soft_diameter = torch.sqrt(max_x**2 + max_y**2)  # Shape (B,)
        return soft_diameter

    def forward(self, inputs, targets):
        """ Compute differentiable loss as absolute diameter difference. """
        inputs = torch.sigmoid(inputs)  # Ensure values are in (0,1) range

        # Extract only the segmentation mask (assumes first channel contains mask)
        inputs = inputs[:, 0, :, :, :]  # Shape (16, 96, 96, 32)

        # Compute predicted diameter
        pred_diameter = self.compute_diameter(inputs)  # Shape (16,)

        # Compute loss (absolute difference with targets)
        loss = torch.abs(pred_diameter - targets).mean()
        return loss
