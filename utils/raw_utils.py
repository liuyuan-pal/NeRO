import torch
import numpy as np

def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps)**(12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError

def reshape_quads(*planes):
    """Reshape pixels from four input images to make tiled 2x2 quads."""
    planes = np.stack(planes, -1)
    shape = planes.shape[:-1]
    # Create [2, 2] arrays out of 4 channels.
    zup = planes.reshape(shape + (2, 2,))
    # Transpose so that x-axis dimensions come before y-axis dimensions.
    zup = np.transpose(zup, (0, 2, 1, 3))
    # Reshape to 2D.
    zup = zup.reshape((shape[0] * 2, shape[1] * 2))
    return zup

def bilinear_upsample(z):
    """2x bilinear image upsample."""
    # Using np.roll makes the right and bottom edges wrap around. The raw image
    # data has a few garbage columns/rows at the edges that must be discarded
    # anyway, so this does not matter in practice.
    # Horizontally interpolated values.
    zx = .5 * (z + np.roll(z, -1, axis=-1))
    # Vertically interpolated values.
    zy = .5 * (z + np.roll(z, -1, axis=-2))
    # Diagonally interpolated values.
    zxy = .5 * (zx + np.roll(zx, -1, axis=-2))
    return reshape_quads(z, zx, zy, zxy)

def upsample_green(g1, g2):
    """Special 2x upsample from the two green channels."""
    z = np.zeros_like(g1)
    z = reshape_quads(z, g1, g2, z)
    alt = 0
    # Grab the 4 directly adjacent neighbors in a "cross" pattern.
    for i in range(4):
        axis = -1 - (i // 2)
        roll = -1 + 2 * (i % 2)
        alt = alt + .25 * np.roll(z, roll, axis=axis)
    # For observed pixels, alt = 0, and for unobserved pixels, alt = avg(cross),
    # so alt + z will have every pixel filled in.
    return alt + z

def bilinear_demosaic_raw_nerf(bayer, mode='rggb'):
    if mode=='rggb':
        r, g1, g2, b = [bayer[(i // 2)::2, (i % 2)::2] for i in range(4)]
    elif mode=='bggr':
        b, g1, g2, r = [bayer[(i // 2)::2, (i % 2)::2] for i in range(4)]
    else:
        raise NotImplementedError
    r = bilinear_upsample(r)
    # Flip in x and y before and after calling upsample, as bilinear_upsample
    # assumes that the samples are at the top-left corner of the 2x2 sample.
    b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]
    g = upsample_green(g1, g2)
    rgb = np.stack([r, g, b], -1)
    return rgb

def bilinear_demosaic_simple(bayer, mode='rggb'):
    if mode=='rggb':
        r, g1, g2, b = [bayer[(i // 2)::2, (i % 2)::2] for i in range(4)]
    elif mode=='bggr':
        b, g1, g2, r = [bayer[(i // 2)::2, (i % 2)::2] for i in range(4)]
    else:
        raise NotImplementedError
    r = bilinear_upsample(r)
    b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]
    g1 = bilinear_upsample(g1)
    g2 = bilinear_upsample(g2[::-1, ::-1])[::-1, ::-1]
    rgbg = np.stack([r, g1, b, g2], -1)
    return rgbg