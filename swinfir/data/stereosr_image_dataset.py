import random
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from swinfir.data.data_util import four_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class StereoSRImageDataset(data.Dataset):

    def __init__(self, opt):
        super(StereoSRImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.use_mixup = opt['use_mixup'] if 'use_mixup' in opt else None
        self.use_channelshuffle = opt['use_channelshuffle'] if 'use_channelshuffle' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = four_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq_l', 'gt_l'],
                                                        self.opt['meta_info_file'], self.filename_tmpl)
        else:
            assert False, "Please load data via meta_info_file!!!"

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path_l = self.paths[index]['gt_l_path']
        img_bytes = self.file_client.get(gt_path_l, 'gt_l')
        img_gt_l = imfrombytes(img_bytes, float32=True)

        gt_path_r = self.paths[index]['gt_r_path']
        img_bytes = self.file_client.get(gt_path_r, 'gt_r')
        img_gt_r = imfrombytes(img_bytes, float32=True)

        lq_path_l = self.paths[index]['lq_l_path']
        img_bytes = self.file_client.get(lq_path_l, 'lq_l')
        img_lq_l = imfrombytes(img_bytes, float32=True)

        lq_path_r = self.paths[index]['lq_r_path']
        img_bytes = self.file_client.get(lq_path_r, 'lq_r')
        img_lq_r = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop([img_gt_l, img_gt_r], [img_lq_l, img_lq_r], gt_size, scale, gt_path_l)
            # flip, rotation
            img_gt_l, img_gt_r, img_lq_l, img_lq_r = augment([img_gt[0], img_gt[1], img_lq[0], img_lq[1]],
                                                             self.opt['use_hflip'], self.opt['use_rot'],
                                                             self.opt['use_vflip'])

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt_l = img_gt_l[0:img_lq_l.shape[0] * scale, 0:img_lq_l.shape[1] * scale, :]
            img_gt_r = img_gt_r[0:img_lq_r.shape[0] * scale, 0:img_lq_r.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_l, img_gt_r, img_lq_l, img_lq_r = img2tensor([img_gt_l, img_gt_r, img_lq_l, img_lq_r],
                                                            bgr2rgb=True, float32=True)

        if self.use_channelshuffle and self.opt['phase'] == 'train':
            perm = np.array([0, 1, 2]) if random.random() < 0.7 else np.random.permutation(3)
            img_lq_l = img_lq_l[perm, :, :]
            img_lq_r = img_lq_r[perm, :, :]
            img_gt_l = img_gt_l[perm, :, :]
            img_gt_r = img_gt_r[perm, :, :]

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt_l, self.mean, self.std, inplace=True)
            normalize(img_gt_r, self.mean, self.std, inplace=True)
            normalize(img_lq_l, self.mean, self.std, inplace=True)
            normalize(img_lq_r, self.mean, self.std, inplace=True)

        img_lq = np.concatenate((img_lq_l, img_lq_r), axis=0)
        img_gt = np.concatenate((img_gt_l, img_gt_r), axis=0)

        if self.opt['phase'] is not 'train':
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path_l, 'gt_path': gt_path_l}

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path_l, 'gt_path': gt_path_l, 'phase': self.opt['phase'],
                'use_mixup': self.opt['use_mixup']}

    def __len__(self):
        return len(self.paths)
