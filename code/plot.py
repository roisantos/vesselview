import torch
from torchviz import make_dot
from models.roinet import RoiNet, ResidualBlock  # Import your model and blocks

# Instantiate the model with the desired parameters
model = RoiNet(
    ch_in=3,
    ch_out=1,
    ls_mid_ch=[32, 64, 128, 128, 64, 32],
    k_size=9,
    cls_init_block=ResidualBlock,
    cls_conv_block=ResidualBlock
)
model.eval()

# Create a dummy input tensor (for example, 1 image of size 256x256 with 3 channels)
x = torch.randn(1, 3, 256, 256)

# Forward pass through the model to get the output
y = model(x)

# Generate the computational graph and save it as a PNG file
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("RoiNet", format="png")
