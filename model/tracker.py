import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

from lib.image import imwrite_indexed
from lib.utils import AverageMeter, ConvRelu, MultiReceptiveConv
from .augmenter import ImageAugmenter
from .discriminator import Discriminator
from .seg_network import SegNetwork
from .template_matching import LongtermTemplate
from .GumbelModule import GumbleSoftmax

from time import time
from PIL import Image



class TargetObject:

    def __init__(self, obj_id, disc_params, **kwargs):

        self.object_id = obj_id
        self.discriminator = Discriminator(**disc_params)
        self.disc_layer = disc_params.layer
        self.start_frame = None
        self.start_mask = None
        self.index = -1

        for key, val in kwargs.items():
            setattr(self, key, val)

    def initialize(self, ft, mask):
        self.discriminator.init(ft[self.disc_layer], mask)

    def classify(self, ft):
        return self.discriminator.apply(ft)


class Tracker(nn.Module):

    def __init__(self, augmenter: ImageAugmenter, feature_extractor, disc_params, Tmatching:LongtermTemplate,
                 Tmat_params, refiner: SegNetwork, device):

        super().__init__()

        self.augmenter = augmenter
        self.augment = augmenter.augment_first_frame
        self.disc_params = disc_params
        self.feature_extractor = feature_extractor

        self.refiner = refiner
        for m in self.refiner.parameters():
            m.requires_grad_(False)
        self.refiner.eval()

        self.device = device

        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0

        self.Tmat_params = Tmat_params
        self.gateF = self.Tmat_params.gateF
        if self.gateF:
            fc1 = nn.Conv2d(Tmat_params.Tmat_out, 16, kernel_size=3, stride=2, padding=1)
            fc1bn = nn.BatchNorm2d(16)
            fc2 = nn.Conv2d(16, 2, kernel_size=3, stride=2, padding=0)
            # initialize the bias of the last fc for
            # initial opening rate of the gate of about 85%
            # fc2.bias.data[0] = 0.1
            # fc2.bias.data[1] = 2
            layers = []
            layers.append(torch.nn.AdaptiveMaxPool2d(7))
            layers.append(fc1)
            layers.append(fc1bn)
            layers.append(torch.nn.AdaptiveMaxPool2d(3))
            layers.append(fc2)
            self.convGS = torch.nn.Sequential(*layers)
            self.gSelect = GumbleSoftmax(hard=True)
        for m in self.convGS.parameters():
            m.requires_grad_(False)
        self.convGS.eval()

        self.Tmatching = Tmatching
        for m in self.Tmatching.parameters():
            m.requires_grad_(False)
        self.Tmatching.eval()

        self.Convert_Diff = nn.Conv2d(Tmat_params.Tmat_out, 1, 3, 2, 1)
        for m in self.Convert_Diff.parameters():
            m.requires_grad_(False)
        self.Convert_Diff.eval()

    def clear(self):
        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0
        torch.cuda.empty_cache()


    def run_dataset(self, dataset, out_path, speedrun=False, restart=None, this_th=0.7):
        """
        :param dataset:   Dataset to work with (See datasets.py)
        :param out_path:  Root path for storing label images. Sequences of label pngs will be created in subdirectories.
        :param speedrun:  [Optional] Whether or not to warm up Pytorch when measuring the run time. Default: False
        :param restart:   [Optional] Name of sequence to restart from. Useful for debugging. Default: None
        """
        out_path.mkdir(exist_ok=True, parents=True)

        dset_fps = AverageMeter()

        print('Evaluating', dataset.name)
        avg_open, N =0, 0
        restarted = False
        for sequence in dataset:
            if restart is not None and not restarted:
                if sequence.name != restart:
                    continue
                restarted = True

            # We preload data as we cannot both read from disk and upload to the GPU in the background,
            # which would be a reasonable thing to do. However, in PyTorch, it is hard or impossible
            # to upload images to the GPU in a data loader running as a separate process.
            sequence.preload(self.device)
            self.clear()  # Mitigate out-of-memory that may occur on some YouTubeVOS sequences on 11GB devices.
            outputs, seq_fps, count_gate,scores = self.run_sequence(sequence, speedrun, this_th=this_th)
            dset_fps.update(seq_fps)

            dst = out_path / sequence.name
            dst.mkdir(exist_ok=True)
            idx=0
            for lb, f in zip(outputs, sequence.frame_names):
                imwrite_indexed(dst / (f + ".png"), lb)
                # if idx >0:
                #     for obj in range(len(scores[idx-1])):
                #         this_s = scores[idx-1][obj]
                #         im = Image.fromarray(this_s, 'P')
                #         im.save(dst / (f + "_" +str(obj)+".png"))

                idx+=1
            for i in range(len(count_gate)):
                avg_open+=count_gate[i]
                N+=1


        print("Average frame rate: %.2f fps" % dset_fps.avg)
        print("Total obj {} reuse Rate {} from {}".format(N, avg_open / N, avg_open))
        return dset_fps.avg, avg_open / N

    def run_sequence(self, sequence, speedrun=False, this_th=0.3):
        """
        :param sequence:  FileSequence to run.
        :param speedrun:  Only for DAVIS 2016: If True, let pytorch initialize its buffers in advance
                          to not incorrectly measure the memory allocation time in the first frame.
        :return:
        """

        self.eval()
        self.object_ids = sequence.obj_ids
        self.current_frame = 0
        self.targets = dict()

        self.Templates = dict()
        self.gate_result= dict()
        self.prev_score= dict()
        self.prevRefine = dict()

        N = 0

        object_ids = torch.tensor([0] + sequence.obj_ids, dtype=torch.uint8, device=self.device)  # Mask -> labels LUT
        count_gate =dict()
        count_frame = dict()
        if speedrun:
            image, labels, obj_ids = sequence[0]
            image = image.to(self.device)
            labels = labels.to(self.device)
            self.initialize(image, labels, sequence.obj_ids)  # Assume DAVIS 2016
            self.track(image, this_th)
            torch.cuda.synchronize()
            self.targets = dict()

            self.Templates = dict()
            self.gate_result = dict()
            self.prev_score = dict()
            self.prevRefine=dict()

        outputs = []
        t0 = time()
        out_score=[]
        for i, (image, labels, new_objects) in tqdm.tqdm(enumerate(sequence), desc=sequence.name, total=len(sequence), unit='frames'):

            old_objects = set(self.targets.keys())

            image = image.to(self.device)
            if len(new_objects) > 0:
                labels = labels.to(self.device)
                self.initialize(image, labels, new_objects)

            if len(old_objects) > 0:
                _, scores= self.track(image, this_th)
                out_score.append(scores)

                masks = self.current_masks
                if len(sequence.obj_ids) == 1:
                    labels = object_ids[(masks[1:2] > 0.5).long()]
                else:
                    masks = torch.clamp(masks, 1e-7, 1 - 1e-7)
                    masks[0:1] = torch.min((1 - masks[1:]), dim=0, keepdim=True)[0]  # background activation
                    segs = F.softmax(masks / (1 - masks), dim=0)  # s = one-hot encoded object activations
                    labels = object_ids[segs.argmax(dim=0)]

            if isinstance(labels, list) and len(labels) == 0:  # No objects yet
                labels = image.new_zeros(1, *image.shape[-2:])

            for k, v in self.gate_result.items():
                try:
                    count_gate[k] = count_gate[k] +v
                    count_frame[k] = count_frame[k]+1
                except:
                    count_gate[k] = v
                    count_frame[k] =1


            outputs.append(labels)
            self.current_frame += 1
            N += 1

            torch.cuda.synchronize()
        T = time() - t0
        fps = N / T
        avg_gate=[]
        for k, v in count_gate.items():
            print("{} th obj open percent {}".format(k, count_gate[k] / count_frame[k]))
            avg_gate.append(count_gate[k] / count_frame[k])

        return outputs, fps, avg_gate,out_score

    def initialize(self, image, labels, new_objects):

        self.current_masks = torch.zeros((len(self.targets) + len(new_objects) + 1, *image.shape[-2:]),
                                         device=self.device)
        with torch.no_grad():
            ft = self.feature_extractor(image)
        ft8 = ft[self.Tmat_params.layer]
        ft16 = ft[self.disc_params.layer]
        _, _, H, W = ft8.size()
        self.H8, self.W8 = H, W

        for obj_id in new_objects:
            # Create target

            mask = (labels == obj_id).byte()
            target = TargetObject(obj_id=obj_id, index=len(self.targets) + 1, disc_params=self.disc_params,
                                  start_frame=self.current_frame, start_mask=mask)
            self.targets[obj_id] = target
            if obj_id != target.index:
                print("obj_id: {} , target_index: {}".format(obj_id, target.index))
            self.gate_result[obj_id] = 1

            # HACK for debugging
            torch.random.manual_seed(0)
            np.random.seed(0)

            # Augment first image and extract features

            im, msk = self.augment(image, mask)
            with torch.no_grad():
                ft_disc = self.feature_extractor(im)
            target.initialize(ft_disc, msk)

            mask = mask.unsqueeze(0)
            prev_seg8 = torch.nn.Upsample(size=(self.H8, self.W8), mode='nearest')(mask.float())
            self.Templates[obj_id] = self.Tmatching(ft8, prev_seg8, None, 0, mode="M")
            self.current_masks[target.index] = mask
            _, _, H, W = ft[self.disc_params.layer].size()
            self.prev_score[obj_id] = torch.nn.Upsample(size=(H, W), mode='nearest')(mask.float())
            scores = target.classify(ft16)
            prevX = self.refiner.Test_Init(scores, ft)
            self.prevRefine[obj_id] = prevX

        return self.current_masks

    def track(self, image, this_th):

        im_size = image.shape[-2:]
        features = self.feature_extractor.get_F8(image)

        # Classify
        ft8 = features[self.Tmat_params.layer]
        scores=[]
        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame:
                # using gating function for check needs of calculate score map
                PrevScore8 = nn.Upsample(size=(self.H8, self.W8), mode='bilinear')(self.prev_score[obj_id])
                tplS = self.Tmatching(torch.cat([ft8,PrevScore8],dim=1), self.Templates[obj_id], mode="Q")
                fc_f = self.convGS(tplS).squeeze(2).squeeze(2)

                _,gateProb = self.gSelect(fc_f)

                if gateProb[0,0]>this_th:
                    gate=1
                else:
                    gate=0
                self.gate_result[obj_id] = gate

                if gate ==0: # we need to calculate
                    PostFeatures = self.feature_extractor.get_F32(ft8)
                    ft16 = PostFeatures[self.disc_params.layer]
                    s = target.classify(ft16)
                    features.update(PostFeatures)
                    Diff=None
                    self.prev_score[obj_id] =s.clone()
                else: # Do not need to calcualte just translate
                    Diff = self.Convert_Diff(tplS)
                    s = self.prev_score[obj_id] + Diff

                Sout=s.squeeze(0).squeeze(0)
                S_min=torch.min(Sout)
                S_max=torch.max(Sout)
                Sout=((255*(Sout-S_min))/(S_max-S_min)).cpu().byte().numpy()
                scores.append(Sout)

                y, prevX = self.refiner.Test_Forward(s, features,gate, self.prevRefine[obj_id], Diff, im_size)
                self.prevRefine[obj_id] = prevX
                self.current_masks[target.index] = torch.sigmoid(y)

        # Update

        for obj_id, t1 in self.targets.items():
            if t1.start_frame < self.current_frame:
                for obj_id2, t2 in self.targets.items():
                    if obj_id != obj_id2 and t2.start_frame == self.current_frame:
                        self.current_masks[t1.index] *= (1 - t2.start_mask.squeeze(0)).float()

        p = torch.clamp(self.current_masks, 1e-7, 1 - 1e-7)
        p[0:1] = torch.min((1 - p[1:]), dim=0, keepdim=True)[0] # bg prob
        segs = F.softmax(p / (1 - p), dim=0) # prob_cls/!prob_cls
        inds = segs.argmax(dim=0)

        # self.out_buffer = segs * F.one_hot(inds, segs.shape[0]).permute(2, 0, 1)
        for i in range(self.current_masks.shape[0]):
            self.current_masks[i] = segs[i] * (inds == i).float()

        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame and self.disc_params.update_filters and self.gate_result[obj_id] ==0:
                target.discriminator.update(self.current_masks[target.index].unsqueeze(0).unsqueeze(0))


        return self.current_masks, scores


