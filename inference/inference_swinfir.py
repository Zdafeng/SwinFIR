# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from swinfir.archs.swinfir_arch import SwinFIR
from swinfir.archs.hatfir_arch import HATFIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/Set5/LRbicx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SwinFIR/Set5', help='output folder')
    parser.add_argument('--task', type=str, default='SwinFIR', help='SwinFIR, SwinFIR-T, HATFIR, HATFIR-L')
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    parser.add_argument('--training_patch_size', type=int, default=60, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/SwinFIR/SwinFIR_SRx4.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            if 'SwinFIR' in args.task:
                window_size = 12
                _, _, h, w = img.size()
                mod_pad_h = (h // window_size + 1) * window_size - h
                mod_pad_w = (w // window_size + 1) * window_size - w
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + mod_pad_h, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]

                output = model(img)
                output = output[..., :h * args.scale, :w * args.scale]
            elif 'HATFIR' in args.task:
                window_size = 16
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h_old, w_old = img_lq.size()
                if h_old % window_size != 0:
                    mod_pad_h = window_size - h_old % window_size
                if w_old % window_size != 0:
                    mod_pad_w = window_size - w_old % window_size
                img_lq = F.pad(img_lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                output = model(img)
                _, _, h, w = output.size()
                output = output[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}_{args.task}.png'), output)


def define_model(args):
    if args.task == 'SwinFIR':
        model = SwinFIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.training_patch_size,
            window_size=12,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='SFB')

    elif args.task == 'SwinFIR-T':
        model = SwinFIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.training_patch_size,
            window_size=12,
            img_range=1.,
            depths=[6, 5, 5, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='HSFB')

    elif args.task == 'HATFIR':
        model = HATFIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.training_patch_size,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='SFB')

    elif args.task == 'HATFIR-L':
        model = HATFIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.training_patch_size,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='SFB')

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
