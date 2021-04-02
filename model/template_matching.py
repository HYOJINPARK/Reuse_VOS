import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class TemplateUpdate(nn.Module):
    def __init__(self, ch_in, ch_key, ch_val, group_n=4):
        super().__init__()
        self.conv_key = nn.Sequential(
            nn.Conv2d(ch_in, ch_key, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_key, ch_key, 5, 1, 2, groups=ch_key//group_n)
        )
        self.conv_val = nn.Sequential(
            nn.Conv2d(ch_in, ch_val, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_val, ch_val, 5, 1, 2, groups=ch_val//group_n)
        )

    def forward(self, input, mask=None):
        if mask !=None:
            input = torch.cat([input, mask], dim=1)
        val = self.conv_val(input)
        key = self.conv_key(input)
        B, V, H, W = val.shape
        B, K, H, W = key.shape
        val_mm = val.reshape(B, V, -1) #V * HW
        key_mm = key.reshape(B, K, -1).permute((0, 2, 1)).contiguous() # HW K
        out = F.softmax(torch.bmm(val_mm, key_mm).mul(1./math.sqrt(H*W)), dim=2) # V*HW HW K
        return(out)




class TemplateMatching(nn.Module):
    def __init__(self, ch_in, ch_key, ch_val, ch_feat, group_n=4):
        super().__init__()

        self.conv_key = nn.Sequential(
            nn.Conv2d(ch_in, ch_key, 1, 1, 0),
            nn.Conv2d(ch_key, ch_key, 5, 1, 2, groups=ch_key//group_n),
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.conv_feat = nn.Sequential(
            nn.Conv2d(ch_in, ch_feat, 1, 1, 0),
            nn.Conv2d(ch_feat, ch_feat, 5, 1, 2, groups=ch_feat//group_n),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blend = nn.Sequential(
            nn.Conv2d(ch_val+ ch_feat,ch_val+ ch_feat, 5, 1, 2, groups=(ch_val+ ch_feat)//4),
            nn.Conv2d(ch_val+ ch_feat, ch_feat, 1, 1, 0)
        )

    def forward(self, input, template):
        key = self.conv_key(input)
        feat = self.conv_feat(input)

        B, K, H, W = key.shape
        key_mm = key.reshape(B, K, -1)
        similarity = torch.bmm(template, key_mm).reshape(B, -1, H, W) # V * K x K * HW => V * HW
        out = F.relu(self.blend(torch.cat([feat, similarity], dim=1)))
        return(out)

class LongtermTemplate(nn.Module):
    def __init__(self, ch_in, ch_key, ch_val, ch_feat, group_n=4):
        super().__init__()

        self.Tmatching = TemplateMatching( ch_in, ch_key, ch_val, ch_feat, group_n)
        self.Tgenerate = TemplateUpdate(ch_in, ch_key, ch_val, group_n)

    def forward(self, input1, input2=None, input3=None, t = None, mode='Q'):
        if mode =='Q':
            return self.Tmatching(input1, input2)
        elif mode =='M':
            curr_gc = self.Tgenerate(input1, input2)
            if t==0:
                return curr_gc
            else:
                w1 = 0.8*(t/(t+1))
                w2 = 0.2*(1/(t+1))
                return (w1/(w1+w2))*input3 +(w2/(w1+w2))*curr_gc
        else:
            print("Wrong mode {}".format(mode))
            exit(0)

