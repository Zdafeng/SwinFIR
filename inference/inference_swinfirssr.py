# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from swinfir.archs.swinfirssr_arch import SwinFIRSSR, SwinFIRSSR_Local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/Flickr1024/lr_x4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SwinFIRSSR/Set5', help='output folder')
    parser.add_argument('--task', type=str, default='SwinFIRSSR_Local', help='SwinFIRSSR, SwinFIRSSR_Local')
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    parser.add_argument('--training_patch_size', type=int, default=64, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 2, 4')
    parser.add_argument('--model_path', type=str,
                        default='experiments/pretrained_models/SwinFIRSSR/SwinFIRSSR_SSRx4.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        if '_lr1' in path:
            continue
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img_l = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_l = torch.from_numpy(np.transpose(img_l[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_l = img_l.unsqueeze(0).to(device)

        img_r = cv2.imread(path.replace('_lr0', '_lr1'), cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_r = torch.from_numpy(np.transpose(img_r[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_r = img_r.unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            window_size = 16
            _, _, h, w = img.size()
            mod_pad_h = (h // window_size + 1) * window_size - h
            mod_pad_w = (w // window_size + 1) * window_size - w

            img_l = torch.cat([img_l, torch.flip(img_l, [2])], 2)[:, :, :h + mod_pad_h, :]
            img_l = torch.cat([img_l, torch.flip(img_l, [3])], 3)[:, :, :, :w + mod_pad_w]

            img_r = torch.cat([img_r, torch.flip(img_r, [2])], 2)[:, :, :h + mod_pad_h, :]
            img_r = torch.cat([img_r, torch.flip(img_r, [3])], 3)[:, :, :, :w + mod_pad_w]
            img = torch.cat([img_l, img_r], 1)

            output = model(img)
            output_l = output[..., :h * args.scale, :w * args.scale]
            output_r = output[..., :h * args.scale, :w * args.scale]

        # save image
        output_l = output_l.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_r = output_r.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output_l.ndim == 3:
            output_l = np.transpose(output_l[[2, 1, 0], :, :], (1, 2, 0))
            output_r = np.transpose(output_r[[2, 1, 0], :, :], (1, 2, 0))
        output_l = (output_l * 255.0).round().astype(np.uint8)
        output_r = (output_r * 255.0).round().astype(np.uint8)

        imgname_l = imgname.replace('_lr0', '_hr0')
        cv2.imwrite(os.path.join(args.output, f'{imgname_l}_{args.task}.png'), output_l)
        imgname_r = imgname_l.replace('_hr0', '_hr1')
        cv2.imwrite(os.path.join(args.output, f'{imgname_r}_{args.task}.png'), output_r)


def define_model(args):
    if args.task == 'SwinFIRSSR':
        model = SwinFIRSSR(
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

    elif args.task == 'SwinFIRSSR_Local':
        model = SwinFIRSSR_Local(
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

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
