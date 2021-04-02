from collections import defaultdict as ddict
from time import time

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.training_datasets import *
from lib.utils import AverageMeter
from lib.Tensor_logger import Logger
import torchvision
import cv2
import numpy as np
import os

class Trainer:

    def __init__(self, name, model, optimizer, scheduler, dataset, checkpoints_path, log_path,
                 max_epochs, batch_size, num_workers=0, load_latest=True, save_interval=5, Init_file=None,
                 stats_to_print=('stats/loss', 'stats/accuracy', 'stats/lr', 'stats/fcache_hits')):

        self.name = name
        self.model = model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dataset = dataset
        self.checkpoints_path = checkpoints_path / name
        self.checkpoints_path.mkdir(exist_ok=True, parents=True)
        self.log_path = log_path / name
        self.visdom_save = os.path.join(self.log_path, "Visdom")
        if not os.path.isdir(self.visdom_save):
            os.mkdir(self.visdom_save)

        self.log = None

        self.epoch = 0
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_interval = save_interval

        self.stats_to_print = stats_to_print

        self.stats = ddict(AverageMeter)

        if Init_file !="":
            self.model.Initalize_state_dict(torch.load(Init_file, map_location='cpu')['model'])
        else:
            print("No initialized pth")

        if load_latest:
            checkpoints = list(sorted(self.checkpoints_path.glob("%s_ep*.pth" % name)))
            if len(checkpoints) > 0:
                self.load_checkpoint(checkpoints[-1])

    def load_checkpoint(self, file):

        print("Loading checkpoint", file)
        ckpt = torch.load(file, map_location='cpu')
        # assert ckpt['name'] == self.name
        self.epoch = ckpt['epoch']
        print("Starting epoch", self.epoch + 1)
        self.stats = ckpt['stats']
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

    def save_checkpoint(self):

        ckpt = dict(name=self.name,
                    epoch=self.epoch,
                    stats=self.stats,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict())
        torch.save(ckpt, self.checkpoints_path / ("%s_ep%04d.pth" % (self.name, self.epoch)))

    def upload_batch(self, batch):

        def recurse_scan(v):
            if torch.is_tensor(v):
                v = v.to(self.model.device)
            elif isinstance(v, (list, tuple)):
                v = [recurse_scan(x) for x in v]
            elif isinstance(v, dict):
                v = {k: recurse_scan(x) for k, x in v.items()}
            return v

        return recurse_scan(batch)

    def update_stats(self, new_stats, iteration, iters_per_epoch, runtime, do_print=False):

        for k, v in new_stats.items():
            self.stats[k].update(v)

        if not do_print:
            return

        header = "{self.epoch}: {iteration}/{iters_per_epoch}, ".format(
            self=self, iteration=iteration, iters_per_epoch=iters_per_epoch,)
        # sps: samples per second
        if iteration % 100 == 0:
            stats = []
            dec = 4
            for k, v in self.stats.items():
                if k in self.stats_to_print:
                    k = k[6:] if k.startswith("stats/") else k
                    s = '{k}={v.val:.{dec}f} ({v.avg:.{dec}f})'.format(k=k, v=v, dec=dec)
                    stats.append(s)

            print(header + ", ".join(stats))

    def log_stats(self):

        if self.log is None:
            self.log = SummaryWriter(str(self.log_path))

        for k, v in self.stats.items():
            self.log.add_scalar(k, v.avg, self.epoch)

    def visdom_out_1chnn(self, img_list, caption):
        for i in range(len(img_list)):
            grid_outputs = torchvision.utils.make_grid(img_list[i].cpu().data, nrow=img_list[i].size(0),normalize=True)
            # self.my_logger.image_summary(grid_outputs, opts=dict(title=caption[i]))
            save_arr = (255 * grid_outputs[0].numpy())
            cv2.imwrite(self.visdom_save + "/" + caption[i] + ".png", save_arr.astype(np.uint8))

    def visdom_out_RGB(self, img_list, caption):
        b,c,H,W = img_list.size()
        h,w = H//8,W//8
        arr_img= np.zeros([h, w*b,3])

        for i in range(b):
            this_img = np.transpose((img_list[i]).cpu().numpy(),(1,2,0)).astype(np.uint8)
            this_img =cv2.resize(this_img, dsize=(w,h), interpolation=cv2.INTER_AREA)
            arr_img[:,i*w:(i+1)*w] = this_img
        cv2.imwrite(self.visdom_save + "/" + caption + ".png", arr_img.astype(np.uint8))

        # grid_outputs = torchvision.utils.make_grid(tensor_img, nrow=b, normalize=False)
        # self.my_logger.image_summary(grid_outputs, opts=dict(title=caption))

    def train(self, m1,  endEP=120):
        # self.my_logger = Logger(8098, self.log_path)

        for epoch in range(self.epoch + 1, self.max_epochs + 1):

            self.epoch = epoch
            self.stats = ddict(AverageMeter)

            dset = ConcatDataset([eval(cls)(**params) for cls, params in self.dataset])

            loader = DataLoader(dset, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=True, shuffle=True)
            t0 = None
            runtime = AverageMeter()
            if epoch < endEP:
                margin = m1 * (epoch / endEP)
            else:
                margin = m1
            print("this epoch target gate prob is {}".format(margin))


            for i, batch in enumerate(loader, 1):
                t0 = time() if t0 is None else t0  # Ignore loader startup pause

                self.optimizer.zero_grad()
                stats, visdom_img, visdom_gt, visdom_seg, visdom_diff, visdom_Tscore, visdom_Oscore, visdom_Fscore=\
                    self.model(*batch, epoch)
                # self.visdom_out_1chnn([visdom_gt, visdom_seg, visdom_diff, visdom_Tscore, visdom_Oscore, visdom_Fscore],
                #                       ["gt_" + str(epoch), "seg_" + str(epoch), "Diff_" + str(epoch),
                #                        "trans_score_" + str(epoch), "Org_score_" + str(epoch),
                #                        "final_score_" + str(epoch)])
                # self.visdom_out_RGB(visdom_img, "InputRGB_" + str(epoch))

                self.optimizer.step()
                runtime.update(time() - t0)
                t0 = time()

                stats['stats/lr'] = self.scheduler.get_last_lr()[0]

                self.update_stats(stats, i, len(loader), runtime, do_print=True)

            self.scheduler.step()
            self.visdom_out_1chnn([visdom_gt, visdom_seg, visdom_diff, visdom_Tscore, visdom_Oscore, visdom_Fscore],
                                  ["gt_"+str(epoch), "seg_"+str(epoch), "Diff_"+str(epoch),
                                   "trans_score_"+str(epoch), "Org_score_"+str(epoch), "final_score_"+str(epoch)])
            self.visdom_out_RGB(visdom_img, "InputRGB_"+str(epoch))

            if self.epoch % self.save_interval == 0:
                self.save_checkpoint()

            self.log_stats()

        print("%s done" % self.name)
