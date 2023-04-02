"""
Adapted from crowsonkb's jax-wavelets. Apparently it works best with transformer architectures
"""
from einops import rearrange
import pywt
import torch
from torch import nn
from torch.nn import functional as F


def get_filter_bank(wavelet, dtype=torch.float32):
    """Get the filter bank for a given pywavelets wavelet name. See
    https://wavelets.pybytes.com for a list of available wavelets.
    Args:
        wavelet: Name of the wavelet.
        dtype: dtype to cast the filter bank to.
    Returns:
        A JAX array containing the filter bank.
    """
    filt = torch.tensor(pywt.Wavelet(wavelet).filter_bank, dtype)
    # Special case for some bior family wavelets
    if wavelet.startswith("bior") and torch.all(filt[:, 0] == 0):
        filt = filt[:, 1:]
    return filt


def make_2d_kernel(lo, hi):
    """Make a 2D convolution kernel from 1D lowpass and highpass filters."""
    lo = torch.flip(lo)
    hi = torch.flip(hi)
    ll = torch.outer(lo, lo)
    lh = torch.outer(hi, lo)
    hl = torch.outer(lo, hi)
    hh = torch.outer(hi, hi)
    kernel = torch.stack([ll, lh, hl, hh])[:, None]
    return kernel


def make_kernels(filter, channels):
    """Precompute the convolution kernels for the DWT and IDWT for a given number of
    channels.
    Args:
        filter: A JAX array containing the filter bank.
        channels: The number of channels in the itorchut image.
    Returns:
        A tuple of JAX arrays containing the convolution kernels for the DWT and IDWT.
    """
    kernel = torch.zeros(
        (channels * 4, channels, filter.shape[1], filter.shape[1]), filter.dtype
    )
    index_i = torch.repeat(torch.arange(4), channels)
    index_j = torch.tile(torch.arange(channels), 4)
    k_dec = make_2d_kernel(filter[0], filter[1])
    k_rec = make_2d_kernel(filter[2], filter[3])
    kernel_dec = kernel.clone()
    kernel_rec = kernel.clone()
    kernel_dec[index_i * channels + index_j, index_j] = k_dec[index_i, 0]
    kernel_rec[index_i * channels + index_j, index_j] = k_rec[index_i, 0]
    kernel_rec = torch.swapaxes(kernel_rec, 0, 1)
    return kernel_dec, kernel_rec


class WaveletEncode2d(nn.Module):
    def __init__(self, channels, wavelet, levels):
        super().__init__()
        self.wavelet = wavelet
        self.channels = channels
        self.levels = levels
        filt = get_filter_bank(wavelet)
        assert filt.shape[-1] % 2 == 1
        kernel, _ = make_kernels(channels)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        for i in range(self.levels):
            low, rest = x[:, : self.channels], x[:, self.channels :]
            pad = self.kernel.shape[-1] // 2
            low = F.pad(low, (pad, pad), "reflect")
            low = F.conv1d(low, self.kernel, stride=2)
            rest = rearrange(
                rest, "n(hi)(wx)(cd)->nhw(cixd)", i=2, x=2, d=self.channels
            )
            x = torch.cat([low, rest], dim=-1)
        return x


class WaveletDecode2d(nn.Module):
    def __init__(self, channels, wavelet, levels):
        super().__init__()
        self.wavelet = wavelet
        self.channels = channels
        self.levels = levels
        filt = get_filter_bank(wavelet)
        assert filt.shape[-1] % 2 == 1
        kernel = filt[2:, None]
        index_i = torch.repeat_interleave(torch.arange(2), channels)
        index_j = torch.tile(torch.arange(channels), (2,))
        kernel_final = torch.zeros(channels * 2, channels, filt.shape[-1])
        kernel_final[index_i * channels + index_j, index_j] = kernel[index_i, 0]
        self.register_buffer("kernel", kernel_final)

    def forward(self, x):
        for i in range(self.levels):
            shape_orig = x.shape
            low, rest = x[:, : self.channels * 4], x[:, self.channels * 4 :]
            pad = self.kernel.shape[-1] // 2 + 2
            low = rearrange(low, "nhw(xic)->n(hi)(wx)c", i=2, x=2)
            low = F.pad(low, (pad, pad), "reflect")
            low = rearrange(low, "n(hi)(wx)c->nhw(xic)", i=2, x=2)
            low = F.conv_transpose1d(
                low, self.kernel, stride=2, padding=self.kernel.shape[-1] // 2
            )
            crop = (low.shape[1] - shape_orig[1] * 2) // 2
            low = low[
                :, crop : low.shape[1] - crop - 1, crop : low.shape[2] - crop - 1, :
            ]
            rest = rearrange(
                rest, "nhw(cixd)->n(hi)(wx)(cd)", i=2, x=2, d=self.channels
            )
            x = torch.cat([low, rest], dim=-1)
        return x
