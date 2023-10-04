import os.path as osp
import swinfir.archs
import swinfir.data
import swinfir.models
import swinfir.losses
import swinfir.metrics
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
