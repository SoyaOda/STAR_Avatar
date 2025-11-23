"""
Training Script for Shape Estimation Network

Features:
- Adam optimizer with learning rate scheduling
- Mixed precision training (AMP) for faster training
- Model checkpointing
- Training and validation loops
- Loss tracking and logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
import time
from pathlib import Path

from models.shape_estimator import ShapeEstimator
from models.star_layer import STARLayer
from data.synthetic_dataset import SyntheticDataset
from data.augmentation import MultiChannelAugmentation, NoAugmentation
from training.losses import ShapeEstimationLoss


class Trainer:
    """Trainer class for shape estimation network."""

    def __init__(self, args):
        """
        Initialize trainer.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

        print(f"Using device: {self.device}")

        # Create output directories
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        print("\nInitializing ShapeEstimator network...")
        self.model = ShapeEstimator(
            num_betas=args.num_betas,
            num_joints=args.num_joints,
            attr_dim=args.attr_dim,
            use_pretrained=args.use_pretrained
        ).to(self.device)

        print(f"  Total parameters: {self.model.get_num_params():,}")

        # Initialize STAR model for geometric loss
        print("\nInitializing STAR model for geometric loss...")
        self.star_model = STARLayer(
            gender='neutral',
            num_betas=args.num_betas
        ).to(self.device)

        # Initialize loss function
        print("Initializing loss function...")
        self.criterion = ShapeEstimationLoss(
            star_model=self.star_model,
            w_beta=args.w_beta,
            w_T=args.w_T,
            w_geo=args.w_geo,
            use_geometric_loss=args.use_geometric_loss
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Mixed precision training
        self.use_amp = args.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        if self.use_amp:
            print("Mixed precision training (AMP) enabled")

        # Load datasets
        print("\nLoading datasets...")
        self._load_datasets()

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        # Load checkpoint if specified
        if args.resume_checkpoint:
            self._load_checkpoint(args.resume_checkpoint)

    def _load_datasets(self):
        """Load training and validation datasets."""
        # Create augmentation transforms
        train_transform = MultiChannelAugmentation(
            horizontal_flip_prob=0.5,
            rotation_degrees=10,
            scale_range=(0.9, 1.1),
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            apply_photometric=True
        ) if self.args.use_augmentation else NoAugmentation()

        val_transform = NoAugmentation()

        # Load full dataset
        full_dataset = SyntheticDataset(
            data_dir=self.args.data_dir,
            transform=None,  # We'll apply transform in training loop
            use_attributes=self.args.use_attributes,
            image_size=self.args.image_size
        )

        # Split into train and validation
        total_size = len(full_dataset)
        val_size = int(total_size * self.args.val_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Store transforms
        self.train_transform = train_transform
        self.val_transform = val_transform

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"  Train samples: {train_size}")
        print(f"  Val samples: {val_size}")

    def _apply_transform(self, batch, transform):
        """Apply transform to a batch."""
        # Apply transform to front and back inputs
        front_input = torch.stack([transform(x) for x in batch['front_input']])
        back_input = torch.stack([transform(x) for x in batch['back_input']])

        return {
            'front_input': front_input,
            'back_input': back_input,
            'beta_gt': batch['beta_gt'],
            'T_gt': batch['T_gt'],
            'attr_input': batch.get('attr_input', None)
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.criterion.train()

        total_loss = 0.0
        total_L_beta = 0.0
        total_L_T = 0.0
        total_L_geo = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.num_epochs}')

        for batch in pbar:
            # Apply augmentation
            batch = self._apply_transform(batch, self.train_transform)

            # Move to device
            front_input = batch['front_input'].to(self.device)
            back_input = batch['back_input'].to(self.device)
            beta_gt = batch['beta_gt'].to(self.device)
            T_gt = batch['T_gt'].to(self.device)
            attr_input = batch['attr_input'].to(self.device) if batch['attr_input'] is not None else None

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                beta_pred, T_pred = self.model(front_input, back_input, attr_input)
                loss_dict = self.criterion(beta_pred, T_pred, beta_gt, T_gt)
                loss = loss_dict['total']

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_L_beta += loss_dict['L_beta'].item()
            total_L_T += loss_dict['L_T'].item()
            if 'L_geo' in loss_dict:
                total_L_geo += loss_dict['L_geo'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'L_β': f"{loss_dict['L_beta'].item():.4f}",
                'L_T': f"{loss_dict['L_T'].item():.4f}"
            })

        # Average losses
        avg_loss = total_loss / num_batches
        avg_L_beta = total_L_beta / num_batches
        avg_L_T = total_L_T / num_batches
        avg_L_geo = total_L_geo / num_batches if num_batches > 0 else 0

        return {
            'loss': avg_loss,
            'L_beta': avg_L_beta,
            'L_T': avg_L_T,
            'L_geo': avg_L_geo
        }

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        self.criterion.eval()

        total_loss = 0.0
        total_L_beta = 0.0
        total_L_T = 0.0
        total_L_geo = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # No augmentation for validation
                batch = self._apply_transform(batch, self.val_transform)

                # Move to device
                front_input = batch['front_input'].to(self.device)
                back_input = batch['back_input'].to(self.device)
                beta_gt = batch['beta_gt'].to(self.device)
                T_gt = batch['T_gt'].to(self.device)
                attr_input = batch['attr_input'].to(self.device) if batch['attr_input'] is not None else None

                # Forward pass
                beta_pred, T_pred = self.model(front_input, back_input, attr_input)
                loss_dict = self.criterion(beta_pred, T_pred, beta_gt, T_gt)

                # Accumulate losses
                total_loss += loss_dict['total'].item()
                total_L_beta += loss_dict['L_beta'].item()
                total_L_T += loss_dict['L_T'].item()
                if 'L_geo' in loss_dict:
                    total_L_geo += loss_dict['L_geo'].item()
                num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_L_beta = total_L_beta / num_batches
        avg_L_T = total_L_T / num_batches
        avg_L_geo = total_L_geo / num_batches if num_batches > 0 else 0

        return {
            'loss': avg_loss,
            'L_beta': avg_L_beta,
            'L_T': avg_L_T,
            'L_geo': avg_L_geo
        }

    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'args': vars(self.args)
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)

        for epoch in range(self.start_epoch, self.args.num_epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])

            epoch_time = time.time() - start_time

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_metrics['loss']:.6f} | "
                  f"L_β: {train_metrics['L_beta']:.6f} | "
                  f"L_T: {train_metrics['L_T']:.6f} | "
                  f"L_geo: {train_metrics['L_geo']:.6f}")
            print(f"  Val Loss:   {val_metrics['loss']:.6f} | "
                  f"L_β: {val_metrics['L_beta']:.6f} | "
                  f"L_T: {val_metrics['L_T']:.6f} | "
                  f"L_geo: {val_metrics['L_geo']:.6f}")

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, val_metrics['loss'], is_best)

        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train Shape Estimation Network')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='outputs/synthetic_data',
                        help='Path to synthetic data directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size (default: 512)')

    # Model parameters
    parser.add_argument('--num_betas', type=int, default=10,
                        help='Number of shape parameters (default: 10)')
    parser.add_argument('--num_joints', type=int, default=16,
                        help='Number of joint heatmap channels (default: 16)')
    parser.add_argument('--attr_dim', type=int, default=3,
                        help='Dimension of user attributes (default: 3)')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='Use pretrained ResNet18 (default: True)')
    parser.add_argument('--use_attributes', action='store_true', default=True,
                        help='Use user attributes (default: True)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')

    # Loss weights
    parser.add_argument('--w_beta', type=float, default=1.0,
                        help='Weight for shape parameter loss (default: 1.0)')
    parser.add_argument('--w_T', type=float, default=1.0,
                        help='Weight for translation loss (default: 1.0)')
    parser.add_argument('--w_geo', type=float, default=0.1,
                        help='Weight for geometric loss (default: 0.1)')
    parser.add_argument('--use_geometric_loss', action='store_true', default=True,
                        help='Use geometric loss (default: True)')

    # Augmentation
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='Use data augmentation (default: True)')

    # Device and optimization
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA if available (default: True)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints',
                        help='Directory to save checkpoints (default: outputs/checkpoints)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
