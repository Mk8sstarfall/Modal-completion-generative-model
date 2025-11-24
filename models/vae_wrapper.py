"""
VAE wrapper for latent diffusion with multi-modal data.
"""

import torch
import torch.nn as nn
from typing import Union, List, Optional, Dict, Any


def get_vae_channels(vae: nn.Module) -> tuple[int, int]:
    """
    Extract input and output channels from VAE.
    Supports diffusers AutoencoderKL and similar architectures.
    """
    # Try diffusers style config
    if hasattr(vae, 'config'):
        in_ch = getattr(vae.config, 'in_channels', None)
        out_ch = getattr(vae.config, 'out_channels', None)
        if in_ch is not None and out_ch is not None:
            return in_ch, out_ch
    
    # Try direct attributes
    if hasattr(vae, 'in_channels') and hasattr(vae, 'out_channels'):
        return vae.in_channels, vae.out_channels
    
    # Try encoder/decoder conv layers
    if hasattr(vae, 'encoder') and hasattr(vae.encoder, 'conv_in'):
        in_ch = vae.encoder.conv_in.in_channels
    elif hasattr(vae, 'conv_in'):
        in_ch = vae.conv_in.in_channels
    else:
        in_ch = None
        
    if hasattr(vae, 'decoder') and hasattr(vae.decoder, 'conv_out'):
        out_ch = vae.decoder.conv_out.out_channels
    elif hasattr(vae, 'conv_out'):
        out_ch = vae.conv_out.out_channels
    else:
        out_ch = None
    
    if in_ch is not None and out_ch is not None:
        return in_ch, out_ch
    
    raise ValueError(
        "Cannot auto-detect VAE channels. Please provide vae_input_channels "
        "and vae_output_channels manually."
    )


class MultiModalVAE(nn.Module):
    """
    Wrapper for VAE that handles multi-modal data.
    Can use separate VAEs per modality or a shared VAE.
    
    Supports channel expansion/contraction:
    - If data channels < VAE required channels (and divisible), 
      data is repeated to match VAE input during encoding.
    - During decoding, repeated channels are averaged back.
    """
    
    def __init__(self,
                 vae: nn.Module,
                 num_modalities: int,
                 data_channels_per_modality: Union[int, List[int]],
                 latent_channels_per_modality: Union[int, List[int]],
                 share_vae: bool = True,
                 scaling_factor: float = 0.18215,
                 vae_input_channels: Optional[Union[int, List[int]]] = None,
                 vae_output_channels: Optional[Union[int, List[int]]] = None):
        """
        Args:
            vae: VAE model(s) for encoding/decoding
            num_modalities: Number of modalities
            data_channels_per_modality: Channels per modality in data space (actual data)
            latent_channels_per_modality: Channels per modality in latent space
            share_vae: Whether to share VAE across modalities
            scaling_factor: Scaling factor for latent space
            vae_input_channels: Channels that VAE expects as input (per modality).
                               If None, auto-detected from VAE.
            vae_output_channels: Channels that VAE outputs (per modality).
                                If None, auto-detected from VAE.
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        self.share_vae = share_vae
        self.scaling_factor = scaling_factor
        
        # Normalize data_channels_per_modality to list
        if isinstance(data_channels_per_modality, int):
            self._data_channels_list = [data_channels_per_modality] * num_modalities
        else:
            self._data_channels_list = list(data_channels_per_modality)
            
        # Normalize latent_channels_per_modality to list
        if isinstance(latent_channels_per_modality, int):
            self._latent_channels_list = [latent_channels_per_modality] * num_modalities
        else:
            self._latent_channels_list = list(latent_channels_per_modality)
        
        # Setup VAE first (needed for auto-detection)
        if share_vae:
            self.vae = vae
            self._vae_list = [vae] * num_modalities
        else:
            if isinstance(vae, nn.ModuleList):
                self.vaes = vae
                self._vae_list = list(vae)
            else:
                self.vaes = nn.ModuleList([vae for _ in range(num_modalities)])
                self._vae_list = [vae] * num_modalities
        
        # Auto-detect or use provided VAE channels
        self._vae_input_channels_list = []
        self._vae_output_channels_list = []
        
        for i in range(num_modalities):
            current_vae = self._vae_list[i]
            detected_in, detected_out = get_vae_channels(current_vae)
            
            # Input channels
            if vae_input_channels is None:
                self._vae_input_channels_list.append(detected_in)
            elif isinstance(vae_input_channels, int):
                self._vae_input_channels_list.append(vae_input_channels)
            else:
                self._vae_input_channels_list.append(vae_input_channels[i])
            
            # Output channels
            if vae_output_channels is None:
                self._vae_output_channels_list.append(detected_out)
            elif isinstance(vae_output_channels, int):
                self._vae_output_channels_list.append(vae_output_channels)
            else:
                self._vae_output_channels_list.append(vae_output_channels[i])
        
        # Calculate channel expansion factors and validate
        self._channel_expand_factors = []
        for i, (data_ch, vae_ch) in enumerate(zip(self._data_channels_list, 
                                                   self._vae_input_channels_list)):
            if vae_ch < data_ch:
                raise ValueError(
                    f"Modality {i}: VAE input channels ({vae_ch}) cannot be less than "
                    f"data channels ({data_ch})"
                )
            if vae_ch % data_ch != 0:
                raise ValueError(
                    f"Modality {i}: VAE input channels ({vae_ch}) must be divisible by "
                    f"data channels ({data_ch})"
                )
            self._channel_expand_factors.append(vae_ch // data_ch)
        
        # Calculate channel contraction factors for decode and validate
        self._channel_contract_factors = []
        for i, (data_ch, vae_ch) in enumerate(zip(self._data_channels_list,
                                                   self._vae_output_channels_list)):
            if vae_ch < data_ch:
                raise ValueError(
                    f"Modality {i}: VAE output channels ({vae_ch}) cannot be less than "
                    f"data channels ({data_ch})"
                )
            if vae_ch % data_ch != 0:
                raise ValueError(
                    f"Modality {i}: VAE output channels ({vae_ch}) must be divisible by "
                    f"data channels ({data_ch})"
                )
            self._channel_contract_factors.append(vae_ch // data_ch)
        
        # Calculate total channels
        self.total_data_channels = sum(self._data_channels_list)
        self.total_latent_channels = sum(self._latent_channels_list)
        
        # For backward compatibility
        self.data_channels_per_modality = data_channels_per_modality
        self.latent_channels_per_modality = latent_channels_per_modality
    
    def _expand_channels(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Expand channels by repeating."""
        if factor == 1:
            return x
        return x.repeat(1, factor, 1, 1)
    
    def _contract_channels(self, x: torch.Tensor, target_channels: int, factor: int) -> torch.Tensor:
        """Contract channels by averaging repeated groups."""
        if factor == 1:
            return x
        B, C, H, W = x.shape
        x = x.view(B, target_channels, factor, H, W)
        return x.mean(dim=2)
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode data space to latent space.
        
        Args:
            x: Input tensor, shape (B, C_data, H, W)
            
        Returns:
            Latent tensor, shape (B, C_latent, H', W')
        """
        if self.share_vae:
            if self.num_modalities == 1:
                x_expanded = self._expand_channels(x, self._channel_expand_factors[0])
            else:
                modalities = torch.split(x, self._data_channels_list, dim=1)
                expanded = [self._expand_channels(mod, factor) 
                           for mod, factor in zip(modalities, self._channel_expand_factors)]
                x_expanded = torch.cat(expanded, dim=1)
            
            latent = self.vae.encode(x_expanded).latent_dist.sample()
            return latent * self.scaling_factor
        else:
            modalities = torch.split(x, self._data_channels_list, dim=1)
            latents = []
            for i, (mod, vae) in enumerate(zip(modalities, self.vaes)):
                mod_expanded = self._expand_channels(mod, self._channel_expand_factors[i])
                latent = vae.encode(mod_expanded).latent_dist.sample()
                latents.append(latent * self.scaling_factor)
            return torch.cat(latents, dim=1)
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent space to data space.
        
        Args:
            z: Latent tensor, shape (B, C_latent, H', W')
            
        Returns:
            Reconstructed tensor, shape (B, C_data, H, W)
        """
        if self.share_vae:
            decoded = self.vae.decode(z / self.scaling_factor).sample
            
            if self.num_modalities == 1:
                return self._contract_channels(
                    decoded, 
                    self._data_channels_list[0], 
                    self._channel_contract_factors[0]
                )
            else:
                modalities = torch.split(decoded, self._vae_output_channels_list, dim=1)
                contracted = [self._contract_channels(mod, data_ch, factor)
                             for mod, data_ch, factor in zip(modalities, 
                                                              self._data_channels_list,
                                                              self._channel_contract_factors)]
                return torch.cat(contracted, dim=1)
        else:
            modalities = torch.split(z, self._latent_channels_list, dim=1)
            decoded = []
            for i, (mod, vae) in enumerate(zip(modalities, self.vaes)):
                dec = vae.decode(mod / self.scaling_factor).sample
                dec_contracted = self._contract_channels(
                    dec,
                    self._data_channels_list[i],
                    self._channel_contract_factors[i]
                )
                decoded.append(dec_contracted)
            return torch.cat(decoded, dim=1)