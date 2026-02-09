import gdown
import torch
from model import NAFNet
from utils import imread_rgb, img2tensor,tensor2img
from utils import tiled_inference, save_img




gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)

def load_nafnet(weights_path, device='cuda'):
    model = NAFNet(
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2]
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state['params'] if 'params' in state else state)
    model.to(device)
    # model.eval()
    return model


def single_image_inference(
    model,
    input_path,
    output_path,
    tile_size=256,
    overlap=32
):
    img = imread_rgb(input_path)
    inp = img2tensor(img)

    out = tiled_inference(
        model,
        inp,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch=4
    )

    out_img = tensor2img([out])
    save_img(out_img, output_path)

    return img, out_img
