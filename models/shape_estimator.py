"""
Shape Estimation Network for STAR Body Model

ResNet18-based network that estimates body shape parameters (β) and global
translation (T) from multi-channel input images (normals + depth + joints + mask).

Based on the specification in md_files/spec1.md (ポーズ情報対応版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ShapeEstimator(nn.Module):
    """
    Shape estimation network using ResNet18 backbone.

    Input: Front and back views, each with 21 channels:
        - Normal map (3 channels)
        - Depth map (1 channel)
        - Silhouette/Mask (1 channel)
        - Joint heatmaps (16 channels for 16 COCO joints, or configurable)

    Output:
        - β: Shape parameters (num_betas dimensions, default 10)
        - T: Global translation (3 dimensions: Tx, Ty, Tz)

    Optional input:
        - User attributes: height, weight, gender (3 dimensions)
    """

    def __init__(self, num_betas=10, num_joints=16, attr_dim=3, use_pretrained=True):
        """
        Initialize shape estimator network.

        Args:
            num_betas: Number of shape parameters to estimate (default: 10)
            num_joints: Number of joint heatmap channels (default: 16)
            attr_dim: Dimension of user attributes (default: 3 for height/weight/gender)
            use_pretrained: Whether to use ImageNet pretrained ResNet18 (default: True)
        """
        super(ShapeEstimator, self).__init__()

        self.num_betas = num_betas
        self.num_joints = num_joints
        self.attr_dim = attr_dim

        # Calculate total input channels per view
        # Normal(3) + Depth(1) + Mask(1) + Joints(num_joints) = 5 + num_joints
        self.input_channels = 5 + num_joints

        # Load pretrained ResNet18
        resnet18 = models.resnet18(pretrained=use_pretrained)

        # Modify first conv layer to accept our multi-channel input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(input_channels, 64, ...)
        original_conv1 = resnet18.conv1

        # Create new conv1 with correct input channels
        self.conv1 = nn.Conv2d(
            self.input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize new conv1 weights
        # Copy pretrained RGB weights and average them for additional channels
        if use_pretrained:
            with torch.no_grad():
                # Copy RGB weights to first 3 channels
                self.conv1.weight[:, :3, :, :] = original_conv1.weight.clone()

                # For additional channels, use average of RGB weights
                rgb_mean = original_conv1.weight.mean(dim=1, keepdim=True)
                for i in range(3, self.input_channels):
                    self.conv1.weight[:, i:i+1, :, :] = rgb_mean.clone()

        # Use rest of ResNet18 layers (after conv1)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool

        # ResNet18 outputs 512-dim feature vector after avgpool
        feature_dim = 512

        # Concatenated features: front(512) + back(512) + attr(attr_dim)
        combined_dim = feature_dim * 2 + attr_dim

        # Fully connected layers for regression
        self.fc1 = nn.Linear(combined_dim, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_betas + 3)  # β + T

        # Initialize fc layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def extract_features(self, x):
        """
        Extract features from input using ResNet18 backbone.

        Args:
            x: Input tensor [B, C, H, W] where C = input_channels

        Returns:
            Feature vector [B, 512]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 512]

        return x

    def forward(self, front_input, back_input, attr_input=None):
        """
        Forward pass of shape estimator.

        Args:
            front_input: Front view tensor [B, 21, H, W]
            back_input: Back view tensor [B, 21, H, W]
            attr_input: Optional user attributes [B, attr_dim] (height, weight, gender)

        Returns:
            beta_pred: Predicted shape parameters [B, num_betas]
            T_pred: Predicted global translation [B, 3] (Tx, Ty, Tz)
        """
        # Extract features from front and back views (shared weights)
        feat_front = self.extract_features(front_input)  # [B, 512]
        feat_back = self.extract_features(back_input)    # [B, 512]

        # Concatenate front and back features
        feat_combined = torch.cat([feat_front, feat_back], dim=1)  # [B, 1024]

        # Add user attributes if provided, otherwise use zeros
        if attr_input is not None:
            feat_combined = torch.cat([feat_combined, attr_input], dim=1)  # [B, 1024+attr_dim]
        else:
            # Pad with zeros when attributes are not provided
            batch_size = feat_combined.shape[0]
            zero_attr = torch.zeros(batch_size, self.attr_dim, device=feat_combined.device)
            feat_combined = torch.cat([feat_combined, zero_attr], dim=1)  # [B, 1024+attr_dim]

        # Fully connected layers
        hidden = self.fc1(feat_combined)  # [B, 256]
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)  # [B, num_betas + 3]

        # Split output into beta and T
        beta_pred = output[:, :self.num_betas]  # [B, num_betas]
        T_pred_raw = output[:, self.num_betas:]  # [B, 3]

        # Apply softplus to Tz (depth) to ensure non-negative
        # Tz should be positive (camera is in front of person)
        Tx = T_pred_raw[:, 0:1]
        Ty = T_pred_raw[:, 1:2]
        Tz = F.softplus(T_pred_raw[:, 2:3])
        T_pred = torch.cat([Tx, Ty, Tz], dim=1)  # [B, 3]

        return beta_pred, T_pred

    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_network():
    """Test the network architecture."""
    print("="*60)
    print("Testing ShapeEstimator Network")
    print("="*60)

    # Create network
    num_betas = 10
    num_joints = 16
    attr_dim = 3

    model = ShapeEstimator(
        num_betas=num_betas,
        num_joints=num_joints,
        attr_dim=attr_dim,
        use_pretrained=False  # Faster for testing
    )

    print(f"\nNetwork Configuration:")
    print(f"  - Input channels per view: {model.input_channels}")
    print(f"  - Number of beta parameters: {num_betas}")
    print(f"  - Number of joint channels: {num_joints}")
    print(f"  - Attribute dimensions: {attr_dim}")
    print(f"  - Total parameters: {model.get_num_params():,}")

    # Create dummy input
    batch_size = 2
    height, width = 512, 512

    # Front and back views: Normal(3) + Depth(1) + Mask(1) + Joints(16) = 21 channels
    front_input = torch.randn(batch_size, model.input_channels, height, width)
    back_input = torch.randn(batch_size, model.input_channels, height, width)

    # User attributes: [height_ratio, weight_ratio, gender]
    # height_ratio: user_height / 1.7
    # weight_ratio: user_weight / 70.0
    # gender: 0 for male, 1 for female
    attr_input = torch.tensor([
        [1.05, 1.1, 0.0],  # Sample 1: 178cm, 77kg, male
        [0.95, 0.9, 1.0],  # Sample 2: 162cm, 63kg, female
    ])

    print(f"\nInput shapes:")
    print(f"  - Front: {front_input.shape}")
    print(f"  - Back: {back_input.shape}")
    print(f"  - Attributes: {attr_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        beta_pred, T_pred = model(front_input, back_input, attr_input)

    print(f"\nOutput shapes:")
    print(f"  - Beta: {beta_pred.shape}")
    print(f"  - T: {T_pred.shape}")

    print(f"\nSample outputs:")
    print(f"  - Beta[0]: {beta_pred[0].numpy()}")
    print(f"  - T[0]: {T_pred[0].numpy()}")

    # Test without attributes
    print(f"\n\nTesting without attributes:")
    with torch.no_grad():
        beta_pred2, T_pred2 = model(front_input, back_input, attr_input=None)

    print(f"  - Beta shape: {beta_pred2.shape}")
    print(f"  - T shape: {T_pred2.shape}")

    print("\n" + "="*60)
    print("✓ Network architecture test passed!")
    print("="*60)


if __name__ == "__main__":
    test_network()
