import torch
import torch.nn as nn
import numpy as np
from lib.utils import AverageMeter, interpolate, ConvRelu, MultiReceptiveConv

from lib.training_datasets import SampleSpec
from .augmenter import ImageAugmenter
from .discriminator import Discriminator
from .template_matching import LongtermTemplate
from .GumbelModule import GumbleSoftmax
import collections
import torchvision
import cv2

class TargetObject:

    def __init__(self, disc_params, **kwargs):

        self.discriminator = Discriminator(**disc_params)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def initialize(self, ft, mask):
        self.discriminator.init(ft[self.discriminator.layer], mask)

    def initialize_pretrained(self, state_dict):

        self.discriminator.load_state_dict(state_dict)
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        self.discriminator.eval()

    def get_state_dict(self):
        return self.discriminator.state_dict()

    def classify(self, ft):
        return self.discriminator.apply(ft)

class TrainerModel(nn.Module):

    def __init__(self, augmenter, feature_extractor, disc_params, Tmatching, Tmat_params,  seg_network, batch_size=0,
                 tmodel_cache=None, device=None):

        super().__init__()

        self.augmenter = augmenter
        self.augment = augmenter.augment_first_frame
        self.tmodels = [TargetObject(disc_params) for _ in range(batch_size)]

        self.Tmat_params = Tmat_params
        self.endEP = Tmat_params.endEP
        self.m1 =  Tmat_params.m1
        self.m2 =  Tmat_params.m2

        self.gateF = self.Tmat_params.gateF
        if self.gateF:
            fc1 = nn.Conv2d(Tmat_params.Tmat_out, 16, kernel_size=3, stride=2, padding=1)
            fc1bn = nn.BatchNorm2d(16)
            fc2 = nn.Conv2d(16, 2, kernel_size=3, stride=2, padding=0)
            # initialize the bias of the last fc for
            # initial opening rate of the gate of about 85%
            fc2.bias.data[0] = 0.1
            fc2.bias.data[1] = 2
            layers = []
            layers.append(torch.nn.AdaptiveMaxPool2d(7))
            layers.append(fc1)
            layers.append(fc1bn)
            layers.append(torch.nn.AdaptiveMaxPool2d(3))
            layers.append(fc2)
            self.convGS = torch.nn.Sequential(*layers)
            self.gSelect = GumbleSoftmax(hard=True)

        self.Tmatching = Tmatching
        self.Convert_Diff = nn.Conv2d(Tmat_params.Tmat_out, 1, 3, 2, 1)

        self.feature_extractor = feature_extractor
        self.refiner = seg_network
        self.tmodel_cache = tmodel_cache
        self.device = device

        self.compute_loss = nn.BCELoss()
        self.compute_accuracy = self.intersection_over_union
        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()

        self.scores = None
        self.ft_channels = None

    def load_state_dict(self, state_dict):

        convGS_dict = dict()
        Tmatching_dict = dict()
        Convert_Diff_dict = dict()
        refiner_dict = dict()

        for k, v in state_dict.items():
            if k.startswith("convGS.") :
                k = k[len("convGS."):]
                convGS_dict[k] = v
            elif k.startswith("Tmatching.") :
                k = k[len("Tmatching."):]
                Tmatching_dict[k] = v
            elif k.startswith("Convert_Diff.") :
                k = k[len("Convert_Diff."):]
                Convert_Diff_dict[k] = v
            elif k.startswith("refiner.") :
                k = k[len("refiner."):]
                refiner_dict[k] = v
            else:
                print("{} wrong state dict ".format(k))

        self.convGS.load_state_dict(convGS_dict)
        self.Tmatching.load_state_dict(Tmatching_dict)
        self.Convert_Diff.load_state_dict(Convert_Diff_dict)
        self.refiner.load_state_dict(refiner_dict)

    def Initalize_state_dict(self, state_dict):
        refiner_dict = dict()

        for k, v in state_dict.items():
            if k.startswith("refiner."):
                k = k[len("refiner."):]
                refiner_dict[k] = v

        self.refiner.load_state_dict(refiner_dict, strict=False)
        print("Load pretrained model")

    def state_dict(self):
        merge_dict = collections.OrderedDict(list(self.convGS.state_dict(prefix="convGS.").items())
                                             + list(self.Tmatching.state_dict(prefix="Tmatching.").items())
                                             + list(self.Convert_Diff.state_dict(prefix="Convert_Diff.").items())
                                             + list(self.refiner.state_dict(prefix="refiner.").items()))
        return merge_dict
        # return self.refiner.state_dict(prefix="refiner.")

    def intersection_over_union(self, pred, gt):

        pred = (pred > 0.5).float()
        gt = (gt > 0.5).float()

        intersection = pred * gt
        i = intersection.sum(dim=-2).sum(dim=-1)
        union = ((pred + gt) > 0.5).float()
        u = union.sum(dim=-2).sum(dim=-1)
        iou = i / u

        iou[torch.isinf(iou)] = 0.0
        iou[torch.isnan(iou)] = 1.0

        return iou

    def forward(self, images, labels, meta, epoch):

        specs = SampleSpec.from_encoded(meta)

        LossSeg = AverageMeter()
        LossDiff = AverageMeter()
        ReuseRate = AverageMeter()
        LossGProb = AverageMeter()

        iter_acc = 0
        n = 0

        cache_hits, Template, prevSeg8, PrevScore16, prevX, PrevFeature = self._initialize(images[0], labels[0], specs)

        #prepare visdom
        b,_, H,W = labels[0].size()
        _,_,h16,w16 = PrevScore16.size()
        Reuse_gates = torch.zeros((b,1)).to(self.device)
        visdom_gt = torch.zeros(len(images) - 1, 1, h16, w16)
        visdom_seg = torch.zeros(len(images) - 1, 1, h16, w16)
        visdom_img = torch.zeros(len(images) - 1, 3,images[0].size(2), images[0].size(3))
        visdom_diff = torch.zeros(len(images) - 1, 1, h16, w16)
        visdom_Tscore = torch.zeros((len(images) - 1, 1, h16, w16))
        visdom_Oscore = torch.zeros((len(images) - 1, 1, h16, w16))
        visdom_Fscore = torch.zeros((len(images) - 1, 1, h16, w16))
        Prev_label = labels[0].to(self.device)

        for i in range(1, len(images)):

            curr_label= labels[i].to(self.device)

            seg, PrevFeature, CurrScore16, prevX, Diff, trans_score, reuse_gate, org_score, final_scores, gate_prob \
                = self._forward(images[i].to(self.device), Template, prevSeg8, PrevScore16, prevX, PrevFeature)
            Reuse_gates+=reuse_gate
            union = ((curr_label + Prev_label.to(self.device)) > 0).float()
            intersect = ((curr_label + Prev_label.to(self.device)) > 1).float()
            gt_sameRatio = torch.sum(intersect.view(b, -1), dim=1) / torch.sum(union.view(b, -1), dim=1)
            gt_sameRatio[gt_sameRatio != gt_sameRatio] = 0
            if epoch < self.endEP:
                margin = self.m1 * (epoch / self.endEP)
            else:
                margin = self.m1
            gt_sameRatio.clamp(margin)
            gt_TargetP = torch.cat([gt_sameRatio.unsqueeze(1), (1 - gt_sameRatio).unsqueeze(1)], dim=1)
            loss_Gprob = self.L2_loss(nn.ReLU()(torch.abs(gate_prob - gt_TargetP) - self.m2),
                                      torch.zeros_like(gt_TargetP))

            # loss for segmentation
            y = curr_label.float()
            acc = self.compute_accuracy(seg.detach(), y)
            loss_seg = self.compute_loss(seg, y)

            # loss for translated score
            Gt_t = torch.nn.Upsample(size=(h16,w16), mode='bilinear')(curr_label.float())
            loss_diff = self.L2_loss(Diff,Gt_t-PrevScore16 )

            loss = loss_seg + loss_diff + loss_Gprob #+ loss_PrevSeg
            loss.backward(retain_graph=True)

            # out stat and visdom
            LossSeg.update(loss_seg.item())
            LossDiff.update(loss_diff.item())
            # LossPrevSeg.update(loss_PrevSeg.item())
            LossGProb.update(loss_Gprob.item())

            iter_acc += acc.mean().cpu().numpy()
            n += 1
            visdom_img[i - 1] = (images[i][0].to(self.device)).detach().clone()
            visdom_gt[i - 1] = torch.nn.Upsample(size=(h16,w16), mode='bilinear')(curr_label[0].float().unsqueeze(0)).squeeze(0)
            visdom_seg[i - 1] = torch.nn.Upsample(size=(h16,w16), mode='bilinear')(seg[0].detach().clone().unsqueeze(0)).squeeze(0)
            visdom_diff[i - 1] = Diff[0].detach().clone()
            visdom_Tscore[i - 1] = trans_score[0].detach().clone()
            visdom_Oscore[i - 1] = org_score[0].detach().clone()
            visdom_Fscore[i - 1] = final_scores[0].detach().clone()
            PrevScore16 = CurrScore16
            Prev_label = (reuse_gate.unsqueeze(2).unsqueeze(2).expand_as(Prev_label)) * Prev_label \
                         + (1-reuse_gate).unsqueeze(2).unsqueeze(2).expand_as(Prev_label) * curr_label
            ReuseRate.update((torch.sum(Reuse_gates)/b).detach().item())


        stats = dict()
        stats['stats/loss_seg'] = LossSeg.avg
        stats['stats/loss_diff'] = LossDiff.avg
        stats['stats/loss_gp'] = LossGProb.avg
        stats['stats/reuseR'] = ReuseRate.avg
        stats['stats/loss'] = LossSeg.avg + LossDiff.avg + LossGProb.avg
        stats['stats/accuracy'] = iter_acc / n
        stats['stats/fcache_hits'] = cache_hits

        return stats, visdom_img, visdom_gt, visdom_seg, visdom_diff, visdom_Tscore, visdom_Oscore, visdom_Fscore

    def _initialize(self, first_image, first_labels, specs):

        cache_hits = 0

        # Augment first image and extract features

        L = self.tmodels[0].discriminator.layer

        N = first_image.shape[0]  # Batch size
        for i in range(N):

            state_dict = None
            if self.tmodel_cache.enable:
                state_dict = self.load_target_model(specs[i], L)

            have_pretrained = (state_dict is not None)

            if not have_pretrained:
                im, lb = self.augment(first_image[i].to(self.device), first_labels[i].to(self.device))
                ft = self.feature_extractor.no_grad_forward(im, output_layers=[L], chunk_size=4)
                self.tmodels[i].initialize(ft, lb)

                if self.tmodel_cache.enable and not self.tmodel_cache.read_only:
                    self.save_target_model(specs[i], L, self.tmodels[i].get_state_dict())

            else:
                if self.ft_channels is None:
                    self.ft_channels = self.feature_extractor.get_out_channels()[L]
                self.tmodels[i].initialize_pretrained(state_dict)
                cache_hits += 1

        ## Template generation
        ft = self.feature_extractor(first_image.to(self.device))
        _, _, H, W = ft[self.Tmat_params.layer].size()
        prevSeg = torch.nn.Upsample(size=(H,W), mode='bilinear')(first_labels.to(self.device).float())
        Templates = self.Tmatching(ft[self.Tmat_params.layer], prevSeg, None, 0, mode="M")
        PrevScore = torch.nn.Upsample(size=(int(np.ceil(H/2)),int(np.ceil(W/2))), mode='bilinear')(first_labels.to(self.device).float())

        scores = []
        for i, tmdl in zip(range(N), self.tmodels):
            x = ft[self.tmodels[0].discriminator.layer][i, None]
            s = tmdl.classify(x)
            scores.append(s)
        scores = torch.cat(scores, dim=0)

        prevX = self.refiner(scores, ft, None, None, None, first_image.shape, scores)

        return cache_hits, Templates, prevSeg, PrevScore, prevX, ft

    def _forward(self, image, Template, prevSeg8, PrevScore16, prevX, PrevFeature=None):

        batch_size = image.shape[0]
        features = self.feature_extractor(image)

        # calculated diff map from previous score map for gating function
        ft8 = features[self.Tmat_params.layer]
        PrevScore8 = nn.Upsample(size=(ft8.size(2), ft8.size(3)), mode='bilinear')(PrevScore16)
        tplS = self.Tmatching(torch.cat([ft8,PrevScore8],dim=1), Template, mode="Q")

        if self.gateF:
            fc_f = self.convGS(tplS).squeeze(2).squeeze(2) # if too much different calculate score again
            gate, gate_prob = self.gSelect(fc_f,force_hard=True)
            gate = gate[:,0].unsqueeze(1)
        else:
            gate = torch.ones(batch_size,1).to(self.device)

        # transform previous score to current score by difference map
        Diff = self.Convert_Diff(tplS) # make diff map
        trans_score = PrevScore16 + Diff
        ##sigmoid(Ps16 +  Diff) = label
        ft = features[self.tmodels[0].discriminator.layer]
        final_scores = []
        org_score=[]
        Mixed_score=[]
        Refine_Feature4 = features['layer4'].clone()
        Refine_Feature5 = features['layer5'].clone()
        # trans_score: transform previous score by diff estimation for loss function
        # Mixed score: disc score + previous score for next frame train and R32, R16
        # Final score : disc score +transform score for refine network R8, R4
        # org_score : score from discrimiannt F for visdom
        for i, tmdl in zip(range(batch_size), self.tmodels):
            if gate[i]==0 :
                x = ft[i, None]
                s = tmdl.classify(x)
                # print("tmdl s size : " + str(s.size()))
                Mixed_score.append(s)
                final_scores.append(s)
                org_score.append(s)

            else:
                Refine_Feature4[i] = PrevFeature['layer4'][i]
                Refine_Feature5[i] = PrevFeature['layer5'][i]
                final_scores.append(trans_score[i].unsqueeze(0))
                Mixed_score.append(PrevScore16[i].unsqueeze(0))
                x = ft[i, None]
                s = tmdl.classify(x)
                org_score.append(s)

        final_scores = torch.cat(final_scores, dim=0)
        Mixed_score = torch.cat(Mixed_score, dim=0)
        org_score = torch.cat(org_score, dim=0)

        Refine_layers={'layer2': features['layer2'], 'layer3': features['layer3'],
                       'layer4': Refine_Feature4, 'layer5': Refine_Feature5, }
        y, prevX, = self.refiner(final_scores, Refine_layers, gate, prevX, Diff, image.shape, Mixed_score)
        y = interpolate(y, image.shape[-2:])

        return torch.sigmoid(y), Refine_layers, Mixed_score, prevX, Diff, trans_score, gate, \
               org_score, final_scores, gate_prob


    def tmodel_filename(self, spec, layer_name):
        return self.tmodel_cache.path / spec.seq_name / ("%05d.%d.%s.pth" % (spec.frame0_id, spec.obj_id, layer_name))

    def load_target_model(self, spec, layer_name):
        fname = self.tmodel_filename(spec, layer_name)
        try:
            state_dict = torch.load(fname, map_location=self.device) if fname.exists() else None
        except Exception as e:
            print("Could not read %s: %s" % (fname, e))
            state_dict = None
        return state_dict

    def save_target_model(self, spec: SampleSpec, layer_name, state_dict):
        fname = self.tmodel_filename(spec, layer_name)
        fname.parent.mkdir(exist_ok=True, parents=True)
        torch.save(state_dict, fname)
