import sys, pathlib
p = str(pathlib.Path(__file__).absolute().parents[1])
if p not in sys.path: sys.path.append(p)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from pathlib import Path
from easydict import EasyDict as edict
from lib.training import Trainer
import argparse
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match nvidia-smi and CUDA device ids
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from model.training_model import TrainerModel
from model.augmenter import ImageAugmenter
from model.feature_extractor import ResnetFeatureExtractor
from model.discriminator import Discriminator
from model.seg_network import SegNetwork
from model.template_matching import LongtermTemplate
from shutil import copyfile


class ModelParameters:

    def __init__(self, name, feature_extractor="resnet101", device="cuda:0", batch_size=None, tmodel_cache_path=None):
        super().__init__()

        self.name = name  # Training session name
        self.device = device
        self.batch_size = batch_size

        # Model parameters

        self.aug_params = edict(

            num_aug=15,
            min_px_count=1,

            fg_aug_params=edict(
                rotation=[5, -5, 10, -10, 20, -20, 30, -30, 45, -45],
                fliplr=[False, False, False, False, True],
                scale=[0.5, 0.7, 1.0, 1.5, 2.0, 2.5],
                skew=[(0.0, 0.0), (0.0, 0.0), (0.1, 0.1)],
                blur_size=[0.0, 0.0, 0.0, 2.0],
                blur_angle=[0, 45, 90, 135],
            ),
            bg_aug_params=edict(
                tcenter=[(0.5, 0.5)],
                rotation=[0, 0, 0],
                fliplr=[False],
                scale=[1.0, 1.0, 1.2],
                skew=[(0.0, 0.0)],
                blur_size=[0.0, 0.0, 1.0, 2.0, 5.0],
                blur_angle=[0, 45, 90, 135],
            ),
        )

        self.disc_params = edict(
            layer="layer4", in_channels=256 if '18' in feature_extractor else 1024, c_channels=32, out_channels=1,
            init_iters=(5, 10, 10, 10, 10), update_iters=(10,), update_filters=True,
            filter_reg=(1e-5, 1e-4), precond=(1e-5, 1e-4), precond_lr=0.1, CG_forgetting_rate=75,
            memory_size=20, train_skipping=8, learning_rate=0.1,
            pixel_weighting=None, device=self.device
        )

        self.refnet_params = edict(
            refinement_layers=["layer5", "layer4", "layer3", "layer2"],
            nchannels=64, use_batch_norm=True
        )
        self.Tmat_params = edict(
            layer="layer3", Tmat_key=64, Tmat_val=64, Tmat_out=64, gateF=True, endEP=60
        )

        self.feature_extractor = feature_extractor

        self.tmodel_cache = edict(
            enable=True,
            read_only=False,
            path=tmodel_cache_path / "{self.feature_extractor}-c{d.c_channels}_gTrue".format(
                self=self, d=self.disc_params)
        )
        print("Get from pth file : {}".format(self.tmodel_cache.path))
    def get_model(self):

        augmenter = ImageAugmenter(self.aug_params)
        extractor = ResnetFeatureExtractor(name=self.feature_extractor).to(self.device)

        p = self.refnet_params
        ft_channels = {L: nch for L, nch in extractor.get_out_channels().items() if L in p.refinement_layers}
        Tmatching = LongtermTemplate(extractor.get_out_channels()[self.Tmat_params.layer] + 1,
                                     self.Tmat_params.Tmat_key, self.Tmat_params.Tmat_val, self.Tmat_params.Tmat_out)
        seg_network = SegNetwork(1, p.nchannels, ft_channels, p.use_batch_norm)
        # seg_network.eval()
        # seg_network.Diff_process.train()
        mdl = TrainerModel(augmenter, extractor, self.disc_params, Tmatching, self.Tmat_params, seg_network,
                           batch_size=self.batch_size, tmodel_cache=self.tmodel_cache, device=self.device)



        mdl = mdl.to(self.device)

        return mdl


if __name__ == '__main__':

    paths = dict(
        dv2017="/media/hyojin/SSD1TB1/Dataset/DAVIS",
        ytvos2018="/media/hyojin/SSD1TB1/Dataset/Youtube-VOS2018",
        checkpoints="./save_dir",  # Checkpoints. Session-name subdirectories will be created here.
        tensorboard="./save_dir",  # Tensorboard logs. Session-name subdirectories will be created here.
        tmcache="./save_dir"     # Cache of pretrained target models, requires 22 GB disk space.
    )
    # paths = dict(
    #     dv2017="~/data/datasets/DAVIS",
    #     ytvos2018="~/data/datasets/YouTubeVOS/2018",
    #     checkpoints="~/workspace/checkpoints",  # Checkpoints. Session-name subdirectories will be created here.
    #     tensorboard="~/workspace/tensorboard",  # Tensorboard logs. Session-name subdirectories will be created here.
    #     tmcache="~/camrdy_ws/tmodels_cache"     # Cache of pretrained target models, requires 22 GB disk space.
    # )
    paths = {k: Path(v).expanduser().resolve() for k, v in paths.items()}

    args_parser = argparse.ArgumentParser(description='Train FRTM')
    args_parser.add_argument('--name', type=str, default='All_RN18_m1000_b8seq6', help='Name of the training session, for logging and saving checkpoints.')
    args_parser.add_argument('--ftext', type=str, default="resnet18", choices=["resnet101", "resnet18"], help='Feature extractor')
    args_parser.add_argument('--dset', type=str, default="all", choices=["all", "yt2018", "dv2017"],
                             help='Training datasets. all = use all data; Both DAVIS 2017 and YouTubeVOS 2018.')
    args_parser.add_argument('--dev', type=str, default="cuda:0", help='Target device to run on, default is cuda:0.')
    args_parser.add_argument('--initPth', type=str, help='Initialization of full model', default="")
    args_parser.add_argument('--marginEp', type=int, default=120, help='Initialization of margin1')
    args_parser.add_argument('--m1', type=float, default=1.0, help='margin 1 for gate prob loss')
    args_parser.add_argument('--m2', type=float, default=0.0, help='margin 2 for gate prob loss')

    args = args_parser.parse_args()
    stats_to_print = ('stats/loss', 'stats/loss_seg', 'stats/loss_diff', 'stats/loss_gp',
                     'stats/accuracy', 'stats/lr', 'stats/reuseR')

    dataset = []
    N_seq=6
    if args.dset in ('all', 'dv2017'):
        dataset.append(('DAVISDataset', edict(dset_path=paths['dv2017'], epoch_repeats=8, sample_size=N_seq)))
    if args.dset in ('all', 'yt2018'):
        dataset.append(('YouTubeVOSDataset', edict(dset_path=paths['ytvos2018'], epoch_samples=4000, min_seq_length=N_seq+2, sample_size=N_seq)))

    params = ModelParameters(args.name, feature_extractor=args.ftext, device=args.dev, tmodel_cache_path=paths['tmcache'], batch_size=8)
    params.Tmat_params.endEP=args.marginEp
    params.Tmat_params.m1 = args.m1
    params.Tmat_params.m2 = args.m2
    model = params.get_model()

    optimizer = torch.optim.Adam(
        [{"params": model.convGS.parameters(), "lr" : 5e-4, "betas" :(0.9, 0.999), "weight_decay" :0, "amsgrad" :True},
         {"params": model.Tmatching.parameters()},
         {"params": model.Convert_Diff.parameters()},
         {"params": model.refiner.parameters()},
         ], lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=127, gamma=0.1)
    trainer = Trainer(args.name, model, optimizer=optimizer, scheduler=scheduler, dataset=dataset, checkpoints_path=paths['checkpoints'],
                      log_path=paths['tensorboard'], max_epochs=260, batch_size=params.batch_size, num_workers=4,
                      load_latest=True, save_interval=1,  Init_file=args.initPth ,stats_to_print=stats_to_print)
    copyfile("train.py", os.path.join(trainer.log_path, "train.py"))

    trainer.train( args.m1,args.marginEp)


