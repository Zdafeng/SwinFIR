import torch
from tqdm import tqdm
from os import path as osp
from collections import OrderedDict

from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from swinfir.models.model_util import mixup


@MODEL_REGISTRY.register()
class SwinFIRSSRModel(SRModel):

    def feed_data(self, data, phase='val'):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'phase' in data and 'use_mixup' in data:
            if data['phase'] == 'train' and data['use_mixup']:
                self.lq, self.gt = mixup(self.lq, self.gt)

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        _, _, h, w = self.lq.size()
        mod_pad_h = (h // window_size + 1) * window_size - h
        mod_pad_w = (w // window_size + 1) * window_size - w
        img = torch.cat([self.lq, torch.flip(self.lq, [2])], 2)[:, :, :h + mod_pad_h, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        self.output = self.output[..., :h * scale, :w * scale]
        self.output_l = self.output[:, :3, :h * scale, :w * scale]
        self.output_r = self.output[:, 3:, :h * scale, :w * scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        metric_data_l = dict()
        metric_data_r = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            sr_img_l = tensor2img([visuals['result_l']])
            sr_img_r = tensor2img([visuals['result_r']])
            metric_data_l['img'] = sr_img_l
            metric_data_r['img'] = sr_img_r
            if 'gt' in visuals:
                gt_img_l = tensor2img([visuals['gt_l']])
                gt_img_r = tensor2img([visuals['gt_r']])
                metric_data_l['img2'] = gt_img_l
                metric_data_r['img2'] = gt_img_r
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.output_l
            del self.output_r
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path_l = osp.join(self.opt['path']['visualization'], img_name,
                                               f'{img_name}_{current_iter}.png')
                else:
                    save_img_path_l = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                save_img_path_l = save_img_path_l.replace('_lr', '_hr')
                save_img_path_r = save_img_path_l.replace('_L', '_R').replace('_hr0', '_hr1')

                imwrite(sr_img_l, save_img_path_l)
                imwrite(sr_img_r, save_img_path_r)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    # self.metric_results[name] += calculate_metric(metric_data, opt_)
                    self.metric_results[name] += calculate_metric(metric_data_l, opt_)
                    self.metric_results[name] += calculate_metric(metric_data_r, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1) * 2
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['result_l'] = self.output_l.detach().cpu()
        out_dict['result_r'] = self.output_r.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
            out_dict['gt_l'] = self.gt[:, :3].detach().cpu()
            out_dict['gt_r'] = self.gt[:, 3:].detach().cpu()
        return out_dict
