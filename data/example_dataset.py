"""
Example custom dataset for medical imaging (MRI modalities).

This is a template showing how to implement a custom dataset.
Replace the placeholder methods with your actual data loading logic.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from .base_dataset import BaseModalDataset


class MedicalMRIDataset(BaseModalDataset):
    """
    Example dataset for multi-modal MRI data.
    
    Expected data structure:
        data_root/
        ├── subject_001/
        │   ├── t1.nii.gz
        │   ├── t2.nii.gz
        │   ├── flair.nii.gz
        │   └── t1ce.nii.gz
        ├── subject_002/
        │   └── ...
        └── ...
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 modality_names: List[str] = None,
                 slice_range: tuple = None,
                 **kwargs):
        """
        Initialize MRI dataset.
        
        Args:
            data_root: Root directory containing subject folders
            split: Dataset split ('train', 'val', 'test')
            modality_names: List of modality filenames (without extension)
            slice_range: Optional (start, end) range of slices to use
            **kwargs: Additional arguments for BaseModalDataset
        """
        if modality_names is None:
            modality_names = ['t1', 't2', 'flair', 't1ce']
        
        self.modality_names = modality_names
        num_modalities = len(modality_names)
        
        super().__init__(
            data_root=data_root,
            num_modalities=num_modalities,
            channels_per_modality=1,  # Single channel per modality
            **kwargs
        )
        
        self.split = split
        self.slice_range = slice_range
        
        # Load data index
        self.samples = self._load_data_index()
        
    def _load_data_index(self) -> List[Dict]:
        """
        Load index of all available samples.
        
        Returns:
            List of dictionaries containing sample information
        """
        data_path = Path(self.data_root)
        samples = []
        
        # TODO: Implement your data indexing logic
        # This is a placeholder - replace with actual implementation
        
        # Example: iterate through subject directories
        for subject_dir in sorted(data_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            # Check if all modalities exist
            modality_files = {}
            for mod_name in self.modality_names:
                mod_file = subject_dir / f"{mod_name}.nii.gz"
                if not mod_file.exists():
                    print(f"Warning: Missing {mod_name} for {subject_dir.name}")
                    break
                modality_files[mod_name] = mod_file
            else:
                # All modalities found
                samples.append({
                    'subject_id': subject_dir.name,
                    'modality_files': modality_files,
                })
        
        # Split data (simple split - consider using proper cross-validation)
        num_samples = len(samples)
        train_split = int(0.8 * num_samples)
        val_split = int(0.9 * num_samples)
        
        if self.split == 'train':
            samples = samples[:train_split]
        elif self.split == 'val':
            samples = samples[train_split:val_split]
        elif self.split == 'test':
            samples = samples[val_split:]
        
        return samples
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single multi-modal sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        sample_info = self.samples[idx]
        
        # TODO: Implement your data loading logic
        # This is a placeholder - replace with actual implementation
        
        modalities = []
        
        for mod_name in self.modality_names:
            # Load modality data
            # Example using nibabel for NIfTI files:
            # import nibabel as nib
            # img = nib.load(sample_info['modality_files'][mod_name])
            # data = img.get_fdata()
            
            # For demonstration, we'll create dummy data
            # REPLACE THIS WITH YOUR ACTUAL LOADING CODE
            data = np.random.randn(1, 256, 256).astype(np.float32)
            
            # Extract slice if specified
            if self.slice_range is not None:
                # For 3D data, you might want to extract 2D slices
                # data = data[:, :, slice_idx]
                pass
            
            # Normalize (example: z-score normalization)
            data = (data - data.mean()) / (data.std() + 1e-8)
            
            # Convert to tensor
            modality_tensor = torch.from_numpy(data)
            modalities.append(modality_tensor)
        
        # Concatenate all modalities along channel dimension
        concatenated = torch.cat(modalities, dim=0)
        
        return {
            'data': concatenated,
            'modalities': modalities,
            'subject_id': sample_info['subject_id'],
            'idx': idx,
        }
    
    def get_modality_info(self) -> Dict[str, any]:
        """Get information about modalities."""
        return {
            'names': self.modality_names,
            'channels': [1] * self.num_modalities,
            'type': 'medical_mri',
            'num_subjects': len(self.samples),
        }


class MultiModalImageDataset(BaseModalDataset):
    """
    Example dataset for multi-modal natural images.
    
    For datasets where different modalities are different views or 
    transformations of the same scene (e.g., RGB, depth, normals, semantic).
    """
    
    def __init__(self,
                 data_root: str,
                 modality_suffixes: List[str] = None,
                 **kwargs):
        """
        Initialize multi-modal image dataset.
        
        Args:
            data_root: Root directory
            modality_suffixes: List of filename suffixes for each modality
                              e.g., ['_rgb.png', '_depth.png', '_normal.png']
            **kwargs: Additional arguments
        """
        if modality_suffixes is None:
            modality_suffixes = ['_rgb.png', '_depth.png', '_normal.png']
        
        self.modality_suffixes = modality_suffixes
        num_modalities = len(modality_suffixes)
        
        # Assume RGB has 3 channels, others have 1 or 3
        # Adjust based on your data
        channels_per_modality = 3  # Adjust as needed
        
        super().__init__(
            data_root=data_root,
            num_modalities=num_modalities,
            channels_per_modality=channels_per_modality,
            **kwargs
        )
        
        self.samples = self._load_data_index()
    
    def _load_data_index(self) -> List[Dict]:
        """Load data index."""
        # TODO: Implement your indexing logic
        # Find all base filenames and check for all modalities
        samples = []
        
        # Placeholder implementation
        data_path = Path(self.data_root)
        # ... your implementation here ...
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load sample."""
        # TODO: Implement your loading logic
        # Use PIL, cv2, or other libraries to load images
        
        # Placeholder
        modalities = []
        for suffix in self.modality_suffixes:
            # Load image
            # from PIL import Image
            # img = Image.open(filepath)
            # data = np.array(img)
            
            # Placeholder data
            data = np.random.randn(3, 256, 256).astype(np.float32)
            modalities.append(torch.from_numpy(data))
        
        concatenated = torch.cat(modalities, dim=0)
        
        return {
            'data': concatenated,
            'modalities': modalities,
            'idx': idx,
        }
    
    def get_modality_info(self) -> Dict[str, any]:
        return {
            'names': ['RGB', 'Depth', 'Normal'],  # Adjust as needed
            'channels': [3, 1, 3],  # Adjust as needed
            'type': 'multi_modal_image',
        }
