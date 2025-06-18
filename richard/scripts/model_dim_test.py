import torch
import torch.nn.functional as F
from richard.src.models.VNet2D import VNet2D
import logging
import argparse
from richard.src.utils.utils import setup_logging

# Create module-level logger
logger = logging.getLogger(__name__)

def test_vnet2d(debug=False, dimensions=(64, 64), batch_size=2, channels=1, device=None):
    """Test function to verify VNet2D model works correctly"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # No need to create a new logger here
    logger.info(f"Running model test with debug={debug}, dimensions={dimensions}, batch_size={batch_size}")
    
    # Create and configure model
    model = VNet2D(in_channels=channels, out_channels=2, debug=debug).to(device)
    
    # Create a random input tensor of specified dimensions
    x = torch.randn(batch_size, channels, *dimensions).to(device)
    
    # Forward pass
    output = model(x)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"VNet2D Model Summary:")
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Test gradient flow
    if debug:
        dummy_target = torch.randn_like(output)
        loss = F.mse_loss(output, dummy_target)
        loss.backward()
        
        model.check_gradient_flow()
    
    return model, output

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Test VNet2D model dimensions')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--width', type=int, default=96, help='Input width')
    parser.add_argument('--height', type=int, default=96, help='Input height')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID (None for CPU)')
    
    args = parser.parse_args()
    
    # Set up logging once at the application entry point
    setup_logging(level=logging.INFO if not args.debug else logging.DEBUG)
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Enable anomaly detection if in debug mode
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    
    # Run test
    test_vnet2d(
        debug=args.debug, 
        dimensions=(args.height, args.width), 
        batch_size=args.batch_size,
        channels=args.channels,
        device=device
    )
    logger.info("Model dimension test completed.")

if __name__ == "__main__":
    main()

