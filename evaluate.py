import os
import sys
from pathlib import Path
from easydict import EasyDict as edict
import argparse

p = str(Path(__file__).absolute().parents[2])
if p not in sys.path:
    sys.path.append(p)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match nvidia-smi and CUDA device ids

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from lib.datasets import DAVISDataset, YouTubeVOSDataset
from model.tracker import Tracker
from model.augmenter import ImageAugmenter
from model.feature_extractor import ResnetFeatureExtractor
from model.seg_network import SegNetwork
from model.template_matching import LongtermTemplate

from lib.evaluation import evaluate_dataset
import time
import numpy as np

class Parameters:

    def __init__(self, weights, fast=False, device="cuda:0"):

        self.device = device
        self.weights = weights
        self.num_aug = 5
        self.train_skipping = 8
        self.learning_rate = 0.1

        # Autodetect the feature extractor

        self.in_channels = weights['refiner.TSE.layer4.reduce.0.weight'].shape[1]
        if self.in_channels == 1024:
            self.feature_extractor = "resnet101"
        elif self.in_channels == 256:
            self.feature_extractor = "resnet18"
        else:
            raise ValueError


        if fast:
            self.init_iters = (5, 10, 10, 10)
            self.update_iters = (5,)
        else:
            self.init_iters = (5, 10, 10, 10, 10)
            self.update_iters = (10,)

        self.aug_params = edict(

            num_aug=self.num_aug,
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
            layer="layer4", in_channels=self.in_channels, c_channels=96, out_channels=1,
            init_iters=self.init_iters, update_iters=self.update_iters,
            memory_size=80, train_skipping=self.train_skipping, learning_rate=self.learning_rate,
            pixel_weighting=dict(method='hinge', tf=0.1),
            filter_reg=(1e-4, 1e-2), precond=(1e-4, 1e-2), precond_lr=0.1, CG_forgetting_rate=750,
            device=self.device, update_filters=True
        )

        self.refnet_params = edict(
            layers=("layer5", "layer4", "layer3", "layer2"),
            nchannels=64, use_batch_norm=True,
        )

        self.Tmat_params = edict(
            layer="layer3", Tmat_key=64, Tmat_val=64, Tmat_out=64, gateF=True
        )

    def get_model(self):

        augmenter = ImageAugmenter(self.aug_params)
        extractor = ResnetFeatureExtractor(self.feature_extractor).to(self.device)
        self.disc_params.in_channels = extractor.get_out_channels()[self.disc_params.layer]
        Tmatching = LongtermTemplate(extractor.get_out_channels()[self.Tmat_params.layer] + 1,
                                     self.Tmat_params.Tmat_key, self.Tmat_params.Tmat_val, self.Tmat_params.Tmat_out)
        p = self.refnet_params
        refinement_layers_channels = {L: nch for L, nch in extractor.get_out_channels().items() if L in p.layers}
        refiner = SegNetwork(self.disc_params.out_channels, p.nchannels, refinement_layers_channels, p.use_batch_norm)

        mdl = Tracker(augmenter, extractor, self.disc_params, Tmatching, self.Tmat_params, refiner, self.device)

        mdl.load_state_dict(self.weights)
        print("Done load model")
        mdl.to(self.device)

        return mdl

    def eval_Flop(self):
        ## calculate complexity
        x = (255*torch.rand( 3, 480, 854)).byte()
        gt = torch.zeros( 1, 480, 854)
        gt[:, 100:200, 100:200] = 1

        gt = gt.byte()


        augmenter = ImageAugmenter(self.aug_params)
        extractor = ResnetFeatureExtractor(self.feature_extractor).to(self.device)
        self.disc_params.in_channels = extractor.get_out_channels()[self.disc_params.layer]
        Tmatching = LongtermTemplate(extractor.get_out_channels()[self.Tmat_params.layer] + 1,
                                     self.Tmat_params.Tmat_key, self.Tmat_params.Tmat_val, self.Tmat_params.Tmat_out)
        p = self.refnet_params
        refinement_layers_channels = {L: nch for L, nch in extractor.get_out_channels().items() if L in p.layers}
        refiner = SegNetwork(self.disc_params.out_channels, p.nchannels, refinement_layers_channels, p.use_batch_norm)

        mdl = MeasureFlopTracker(augmenter, extractor, self.disc_params, Tmatching, self.Tmat_params, refiner, self.device)
        mdl.to(self.device)

        all_init, all_track = 0, 0
        for i in range(11):
            init_t, track_t = mdl(x, gt)
            if i > 0:
                all_init += init_t
                all_track += track_t
                print("{} th init :{} \t track :{}".format(i, init_t, track_t))
        print("total init :{} \t track :{}".format(all_init/10, all_track/10))



if __name__ == '__main__':

    # Edit these paths for your local setup

    paths = dict(
        models=Path(__file__).parent / "save_dir",  # The .pth files should be here
        davis="/media/hyojin/SSD1TB1/Dataset/DAVIS",  # DAVIS dataset root
        yt2018="/media/hyojin/SSD1TB1/Dataset/Youtube-VOS2018",  # YouTubeVOS 2018 root
        output="./results",  # Output path
    )
    # paths = dict(
    #     models=Path(__file__).parent / "weights",  # The .pth files should be here
    #     davis="/mnt/data/camrdy_ws/DAVIS",  # DAVIS dataset root
    #     yt2018="/mnt/data/camrdy_ws/YouTubeVOS/2018",  # YouTubeVOS 2018 root
    #     output="/mnt/data/camrdy_ws/results",  # Output path
    # )

    datasets = dict(
                    dv2016val=(DAVISDataset, dict(path=paths['davis'], year='2016', split='val')),
                    dv2017val=(DAVISDataset, dict(path=paths['davis'], year='2017', split='val')),
                    yt2018jjval=(YouTubeVOSDataset, dict(path=paths['yt2018'], year='2018', split='jjval')),
                    yt2018val=(YouTubeVOSDataset, dict(path=paths['yt2018'], year='2018', split='valid')),
                    yt2018valAll=(YouTubeVOSDataset, dict(path=paths['yt2018'], year='2018', split='valid_all_frames')))

    args_parser = argparse.ArgumentParser(description='Evaluate FRTM on a validation dataset')
    args_parser.add_argument('--model', type=str,default='DAVIS_RN18_m1000.pth', help='name of model weights file')
    args_parser.add_argument('--dset', type=str, default='dv2016val', choices=datasets.keys(), help='Dataset name.')
    args_parser.add_argument('--dev', type=str, default="cuda:1", help='Target device to run on.')
    args_parser.add_argument('--fast', type=bool, default=True, help='Whether to use fewer optimizer steps.')
    args_parser.add_argument('--tau',type=str,default=0.7, help='Threshold for reuse-gate')
    args = args_parser.parse_args()

    model_path = Path(paths['models']).expanduser().resolve() / args.model
    if not model_path.exists():
        print("Model file '%s' not found." % model_path)
        quit(1)
    weights = torch.load(model_path, map_location='cpu')['model']

    dset = datasets[args.dset]
    dset = dset[0](**dset[1])

    ex_name = dset.name + "-" + model_path.stem + "_" + str(args.tau) + ("_fast" if args.fast else "")
    out_path = Path(paths['output']).expanduser().resolve() / ex_name
    out_path.mkdir(exist_ok=True, parents=True)

    # Run the tracker and evaluate the results

    p = Parameters(weights, fast=args.fast, device=args.dev)
    tracker = p.get_model()

    fps, reuseR = tracker.run_dataset(dset, out_path, speedrun=args.dset == 'dv2016val',this_th=args.tau)
    if ('dv' in args.dset )or ('yt2018jjval' == args.dset):
        dset.all_annotations = True

        print("Computing J-scores")
        J=evaluate_dataset(dset, out_path, measure='J')

        print("Computing F-scores")
        F=evaluate_dataset(dset, out_path, measure='F')

        print("Result of : {}".format(args.model))
        print("FPS : {} \t  reuse Rate : {} \t J score :{} \t F scroe : {}".format(fps, reuseR, J, F))
    else:
        print("FPS : {} \t  reuse Rate : {} ".format(fps, reuseR))

    #
