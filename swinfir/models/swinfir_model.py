import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from swinfir.models.model_util import mixup


@MODEL_REGISTRY.register()
class SwinFIRModel(SRModel):

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
