"""
NYU Depth V2 Dataset

Load NYU Depth V2 dataset from local CSV files.
RGB (3 channels) + Depth (1 channel) concatenated along channel dimension.

Usage:
    dataset = NYUDepthV2Dataset(
        data_root='nyu_data',
        split='train',
        target_size=(256, 256),
        normalize=True,
        normalization_range=(-1.0, 1.0)
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import csv
import numpy as np
import torch
from PIL import Image

from .base_dataset import BaseModalDataset


class NYUDepthV2Dataset(BaseModalDataset):
    """
    NYU Depth V2 Dataset
    
    Loads RGB-D data from local CSV files.
    Data is concatenated as [RGB (3ch), Depth (1ch)] = 4 channels total.
    
    Expected directory structure:
        data_root/
        ├── data/
        │   ├── nyu2_train.csv
        │   ├── nyu2_test.csv
        │   ├── nyu2_train/
        │   │   └── ... (RGB .jpg and depth .png files)
        │   └── nyu2_test/
        │       └── ... (RGB and depth .png files)
    """
    
    MODALITIES = ['rgb', 'depth']
    CHANNELS_PER_MODALITY = [3, 1]  # RGB: 3 channels, Depth: 1 channel
    
    def __init__(self,
                 data_root: str = 'nyu_data',
                 split: str = 'train',
                 target_size: Union[int, Tuple[int, int]] = (256, 256),
                 train_ratio: float = 0.9,
                 interpolation: str = 'bilinear',
                 **kwargs):
        """
        Initialize NYU Depth V2 dataset
        
        Args:
            data_root: Root directory containing the data/ folder with CSV files
            split: One of 'train', 'val', 'test'
            target_size: Target image size. Can be int (square) or (H, W) tuple
            train_ratio: Ratio of training CSV to use for train vs val split (default: 0.9)
            interpolation: Interpolation mode for resizing ('bilinear', 'nearest', 'bicubic')
            **kwargs: Additional arguments for BaseModalDataset
        """
        # Handle target_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        # RGB: 3 channels, Depth: 1 channel
        super().__init__(
            data_root=data_root,
            num_modalities=2,
            channels_per_modality=self.CHANNELS_PER_MODALITY,
            **kwargs
        )
        
        self.split = split
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.data_root = Path(data_root)
        
        # Set interpolation mode
        self.interpolation = {
            'bilinear': Image.BILINEAR,
            'nearest': Image.NEAREST,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
        }.get(interpolation.lower(), Image.BILINEAR)
        
        # Load dataset from CSV files
        self._load_local_dataset()
        
        print(f"[NYU Depth V2] Loaded {split} set: {len(self.samples)} samples, "
              f"target size: {target_size}")
    
    def _load_local_dataset(self):
        """Load dataset from local CSV files"""
        
        # Define CSV file paths
        train_csv = self.data_root / 'data' / 'nyu2_train.csv'
        test_csv = self.data_root / 'data' / 'nyu2_test.csv'
        
        # Check if files exist
        if not train_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")
        
        print(f"[NYU Depth V2] Loading from local CSV files...")
        
        # Load CSV files
        train_pairs = self._read_csv(train_csv)
        test_pairs = self._read_csv(test_csv)
        
        # Split training data into train/val
        num_train = len(train_pairs)
        train_end = int(num_train * self.train_ratio)
        
        if self.split == 'train':
            self.samples = train_pairs[:train_end]
        elif self.split == 'val':
            self.samples = train_pairs[train_end:]
        elif self.split == 'test':
            self.samples = test_pairs
        elif self.split == 'all':
            self.samples = train_pairs + test_pairs
        else:
            raise ValueError(
                f"Unknown split: '{self.split}'. "
                f"Use 'train', 'val', 'test', or 'all'"
            )
        
        # Store split info
        self._split_info = {
            'total_train_csv': num_train,
            'total_test_csv': len(test_pairs),
            'train': train_end,
            'val': num_train - train_end,
            'test': len(test_pairs),
        }
    
    def _read_csv(self, csv_path: Path) -> List[Tuple[Path, Path]]:
        """
        Read CSV file and return list of (rgb_path, depth_path) tuples
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of (rgb_path, depth_path) tuples
        """
        pairs = []
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    rgb_path = self.data_root / row[0].strip()
                    depth_path = self.data_root / row[1].strip()
                    pairs.append((rgb_path, depth_path))
        
        return pairs
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.samples)
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single RGB-D sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'data': Concatenated [RGB, Depth] tensor, shape (4, H, W)
                - 'idx': Sample index
                - 'original_size': Original image size (H, W)
                - 'depth_range': (min, max) of original depth values
        """
        rgb_path, depth_path = self.samples[idx]
        
        # ===== Process RGB Image =====
        rgb_image = Image.open(rgb_path)
        
        # Ensure RGB mode
        if rgb_image.mode != 'RGB':
            rgb_image = rgb_image.convert('RGB')
        
        # Store original size
        original_size = (rgb_image.size[1], rgb_image.size[0])  # (H, W)
        
        # Resize RGB
        rgb_resized = rgb_image.resize(
            (self.target_size[1], self.target_size[0]),  # PIL uses (W, H)
            self.interpolation
        )
        
        # Convert to tensor: (3, H, W), range [0, 1]
        rgb_array = np.array(rgb_resized, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        
        # ===== Process Depth Map =====
        depth_image = Image.open(depth_path)
        
        # Convert to numpy array
        depth_array = np.array(depth_image, dtype=np.float32)
        
        # If depth has multiple channels, take the first one
        if depth_array.ndim == 3:
            depth_array = depth_array[:, :, 0]
        
        # Store original depth range
        depth_min_orig = float(depth_array.min())
        depth_max_orig = float(depth_array.max())
        
        # Resize depth
        depth_pil = Image.fromarray(depth_array, mode='F')  # 'F' mode for float32
        depth_resized = depth_pil.resize(
            (self.target_size[1], self.target_size[0]),
            self.interpolation
        )
        depth_array = np.array(depth_resized, dtype=np.float32)
        
        # Normalize depth to [0, 1]
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        if depth_max - depth_min > 1e-6:
            depth_array = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            depth_array = np.zeros_like(depth_array)
        
        # Convert to tensor: (1, H, W)
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
        
        # ===== Concatenate Modalities =====
        # Shape: (4, H, W) = [RGB(3), Depth(1)]
        data_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        
        return {
            'data': data_tensor,
            'idx': idx,
            'original_size': original_size,
            'depth_range': (depth_min_orig, depth_max_orig),
            'rgb_path': str(rgb_path),
            'depth_path': str(depth_path),
        }
    
    def get_modality_info(self) -> Dict[str, any]:
        """Get information about NYU Depth V2 modalities"""
        return {
            'names': self.MODALITIES,
            'channels': self.CHANNELS_PER_MODALITY,
            'type': 'rgbd',
            'dataset': 'NYU Depth V2',
            'source': 'Local CSV files',
            'description': {
                'rgb': 'RGB color image (3 channels, range [0, 1])',
                'depth': 'Depth map (1 channel, normalized to [0, 1])'
            },
            'target_size': self.target_size,
            'original_info': {
                'original_size': (480, 640),
                'depth_unit': 'meters',
                'scene_types': 'indoor (bedroom, kitchen, living room, etc.)'
            }
        }
    
    def get_dataset_statistics(self) -> Dict[str, any]:
        """Get dataset statistics"""
        return {
            'split': self.split,
            'num_samples': len(self.samples),
            'target_size': self.target_size,
            'modalities': self.MODALITIES,
            'channels_per_modality': self.CHANNELS_PER_MODALITY,
            'total_channels': sum(self.CHANNELS_PER_MODALITY),
            'split_info': self._split_info,
        }
    
    def split_modalities(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        Split concatenated data into RGB and Depth
        
        Args:
            data: Concatenated data, shape (4, H, W) or (B, 4, H, W)
            
        Returns:
            List of [rgb_tensor (3, H, W), depth_tensor (1, H, W)]
        """
        if data.dim() == 3:  # Single sample: (4, H, W)
            rgb = data[:3]   # (3, H, W)
            depth = data[3:] # (1, H, W)
        elif data.dim() == 4:  # Batched: (B, 4, H, W)
            rgb = data[:, :3]   # (B, 3, H, W)
            depth = data[:, 3:] # (B, 1, H, W)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {data.dim()}D")
        
        return [rgb, depth]
    
    def combine_modalities(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine RGB and Depth into concatenated tensor
        
        Args:
            modalities: List of [rgb_tensor, depth_tensor]
            
        Returns:
            Concatenated tensor (4, H, W) or (B, 4, H, W)
        """
        rgb, depth = modalities
        channel_dim = 0 if rgb.dim() == 3 else 1
        return torch.cat([rgb, depth], dim=channel_dim)
    
    def visualize_sample(self, idx: int, return_tensors: bool = False) -> Dict[str, np.ndarray]:
        """
        Get a sample for visualization (denormalized)
        
        Args:
            idx: Sample index
            return_tensors: If True, return torch tensors instead of numpy arrays
            
        Returns:
            Dictionary with 'rgb' and 'depth' arrays/tensors
        """
        sample = self[idx]
        data = sample['data']
        
        # Denormalize if needed
        if self.normalize:
            data = self._denormalize(data)
        
        # Split modalities
        rgb, depth = self.split_modalities(data)
        
        if return_tensors:
            return {'rgb': rgb, 'depth': depth}
        else:
            return {
                'rgb': rgb.numpy(),
                'depth': depth.numpy()
            }


def test_nyu_dataset():
    """Test the NYU Depth V2 dataset"""
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing NYU Depth V2 Dataset")
    print("=" * 60)
    
    # Initialize dataset
    dataset = NYUDepthV2Dataset(
        data_root='nyu_data',
        split='train',
        target_size=(256, 256),
        normalize=True,
        normalization_range=(-1.0, 1.0),
    )
    
    # Print dataset information
    print(f"\n[Dataset Info]")
    print(f"  Dataset size: {len(dataset)}")
    
    stats = dataset.get_dataset_statistics()
    print(f"  Target size: {stats['target_size']}")
    print(f"  Modalities: {stats['modalities']}")
    print(f"  Channels: {stats['channels_per_modality']}")
    print(f"  Total channels: {stats['total_channels']}")
    print(f"  Split info: {stats['split_info']}")
    
    modality_info = dataset.get_modality_info()
    print(f"\n[Modality Info]")
    print(f"  Type: {modality_info['type']}")
    print(f"  Source: {modality_info['source']}")
    
    # Load sample
    print(f"\n[Loading Sample 0]")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Data shape: {sample['data'].shape}")
    print(f"  Data range: [{sample['data'].min():.3f}, {sample['data'].max():.3f}]")
    print(f"  Original size: {sample['original_size']}")
    print(f"  Depth range (original): {sample['depth_range']}")
    print(f"  RGB path: {sample['rgb_path']}")
    print(f"  Depth path: {sample['depth_path']}")
    
    # Test split_modalities
    rgb, depth = dataset.split_modalities(sample['data'])
    print(f"\n[Split Modalities]")
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    
    # Test combine_modalities
    recombined = dataset.combine_modalities([rgb, depth])
    print(f"  Recombined shape: {recombined.shape}")
    print(f"  Recombine correct: {torch.allclose(recombined, sample['data'])}")
    
    # Visualize
    print(f"\n[Visualization]")
    vis_data = dataset.visualize_sample(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RGB - need to transpose from (3, H, W) to (H, W, 3)
    rgb_vis = vis_data['rgb']
    rgb_vis = np.transpose(rgb_vis, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
    rgb_vis = np.clip(rgb_vis, 0, 1)
    axes[0].imshow(rgb_vis)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Depth
    depth_vis = vis_data['depth'][0]  # (1, H, W) -> (H, W)
    im = axes[1].imshow(depth_vis, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('nyu_depth_v2_sample.png', dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: nyu_depth_v2_sample.png")
    
    # Test different splits
    print(f"\n[Testing Different Splits]")
    for split in ['train', 'val', 'test']:
        ds = NYUDepthV2Dataset(data_root='nyu_data', split=split, target_size=64)
        print(f"  {split}: {len(ds)} samples")
    
    # Test loading speed
    print(f"\n[Loading Speed Test]")
    import time
    num_samples = min(50, len(dataset))
    start = time.time()
    for i in range(num_samples):
        _ = dataset[i]
    elapsed = time.time() - start
    print(f"  Time to load {num_samples} samples: {elapsed:.3f} seconds")
    print(f"  Average per sample: {elapsed/num_samples*1000:.2f} ms")
    
    # Test batched split_modalities
    print(f"\n[Testing Batched Operations]")
    batch_data = torch.stack([dataset[i]['data'] for i in range(4)])
    print(f"  Batch shape: {batch_data.shape}")
    rgb_batch, depth_batch = dataset.split_modalities(batch_data)
    print(f"  RGB batch shape: {rgb_batch.shape}")
    print(f"  Depth batch shape: {depth_batch.shape}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_nyu_dataset()