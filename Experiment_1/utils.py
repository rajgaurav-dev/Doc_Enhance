import math
import cv2
import torch
import numpy as np
from torchvision.utils import make_grid

# ---------------------------------
# Image I/O utilities
# ---------------------------------
def imread_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_img(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def img2tensor(img):
    """
    img: H x W x C, RGB, uint8
    return: C x H x W, float32 in [0,1]
    """
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return img

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            elif img_np.shape[2] == 3:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


# ---------------------------------
# Tiled inference
# ---------------------------------
@torch.no_grad()
def tiled_inference(
    model,
    img_tensor,          # (C, H, W), float32 [0,1]
    tile_size=256,
    overlap=32,
    tile_batch=4
):
    """
    Runs tiled inference to avoid OOM for large images
    """
    assert img_tensor.dim() == 3, "img_tensor must be (C, H, W)"

    model.eval()
    device = next(model.parameters()).device

    img = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    _, C, H, W = img.shape

    stride = tile_size - overlap
    assert stride > 0, "overlap must be smaller than tile_size"

    output = torch.zeros((1, C, H, W), device=device)
    weight = torch.zeros((1, 1, H, W), device=device)

    tiles = []

    # ---------------------------
    # Create tiles
    # ---------------------------
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)

            y_start = max(y_end - tile_size, 0)
            x_start = max(x_end - tile_size, 0)

            tile = img[:, :, y_start:y_start + tile_size,
                             x_start:x_start + tile_size]

            tiles.append((tile, y_start, x_start))

    # ---------------------------
    # Process tiles in batches
    # ---------------------------
    for i in range(0, len(tiles), tile_batch):
        batch = tiles[i:i + tile_batch]
        batch_tensor = torch.cat([t[0] for t in batch], dim=0)

        preds = model(batch_tensor)

        # Defensive crop (important!)
        preds = preds[..., :tile_size, :tile_size]

        for b_idx, (_, y, x) in enumerate(batch):
            output[:, :, y:y + tile_size, x:x + tile_size] += preds[b_idx:b_idx + 1]
            weight[:, :, y:y + tile_size, x:x + tile_size] += 1.0

    # Avoid divide-by-zero
    weight = torch.clamp(weight, min=1.0)

    return (output / weight).squeeze(0)

