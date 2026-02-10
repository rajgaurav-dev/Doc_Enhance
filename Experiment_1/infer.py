import os
import argparse
import gdown
import torch

from model import NAFNet
from utils import (
    imread_rgb,
    img2tensor,
    tensor2img,
    tiled_inference,
    save_img,
    load_nafnet
)

# -------------------------------------------------
# Default paths
# -------------------------------------------------
DEFAULT_WEIGHTS_DIR = "./pretrained_models"
DEFAULT_WEIGHTS_PATH = os.path.join(
    DEFAULT_WEIGHTS_DIR, "nafnet_sidd.pth"
)
GDRIVE_ID = "14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR"


# -------------------------------------------------
# Download weights if missing
# -------------------------------------------------
def download_weights(weights_path: str):
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    if not os.path.exists(weights_path):
        print("⬇ Downloading pretrained NAFNet weights...")
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_ID}",
            weights_path,
            quiet=False,
        )
        print("✅ Weights downloaded")
    else:
        print("✅ Pretrained weights found")





# -------------------------------------------------
# Inference
# -------------------------------------------------
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    tile_size: int,
    overlap: int,
):
    """
    Run tiled inference on a single image.
    """
    print(f"Reading image: {input_path}")
    img = imread_rgb(input_path)
    inp = img2tensor(img).to(next(model.parameters()).device)

    print("Running tiled inference...")
    out = tiled_inference(
        model=model,
        img_tensor=inp,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch=4,
    )

    out_img = tensor2img(out)
    save_img(out_img, output_path)

    print(f"Output saved to: {output_path}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="NAFNet Document Image Restoration"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Path to save output image",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="Tile size for inference",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Overlap between tiles",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()

    download_weights(args.weights)
    model = load_nafnet(args.weights, args.device)

    run_inference(
        model=model,
        input_path=args.input,
        output_path=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    print("Inference completed successfully")


if __name__ == "__main__":
    main()
