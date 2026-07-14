import torch
import torch.nn as nn
from torch.nn import Upsample
from torch.nn import functional as F
from copy import deepcopy

from ultralytics.nn.modules.block import Conv, C3k2, SPPF, C2PSA
from ultralytics.nn.tasks import BaseModel
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Concat
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import initialize_weights, intersect_dicts

from models.loss import E2EDetectLoss


class DetectionModel(BaseModel):
    def __init__(self, cfg=None, ch=3, nc=80, verbose=True):
        super().__init__()
        self.model = DetNet(nc=nc)
        # hpy config
        ch = 3  
        self.nc = nc
        self.yaml = {'nc': self.nc, 'ch': ch}
        self.names = {i: f'{i}' for i in range(self.nc)}
        self.inplace = True
        self.end2end = False
        # build strides
        m = self.model.head
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                return self.forward(x)  

            m.stride = torch.tensor([
                s / x[0].shape[-2] for x in _forward([(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])[0]
            ])
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        LOGGER.info("")

    def forward(self, x, *args, **kwargs):
        if isinstance(x[0], dict):
            loss_list = self.loss(x, *args, **kwargs)
            loss = torch.mean(torch.cat([l_it[0].unsqueeze(0) for l_it in loss_list], dim=0), dim=0)
            loss_item = torch.mean(torch.cat([l_it[1].unsqueeze(0) for l_it in loss_list], dim=0), dim=0)
            return loss, loss_item
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        return self._predict_once(x, profile, visualize, embed)

    def loss(self, batch, preds=None):
        """
        Computes the loss.
        :param batch: <dict> batch to compute loss on
        :param preds: <torch.Tensor | List[torch.Tensor]> predictions
        """
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        in_x = [(bi['img'], bi['evt']) for bi in batch]
        preds = self.model.forward(in_x) if preds is None else preds
        loss_list = [self.criterion(pi, ti) for pi, ti in zip(preds, batch)]
        return loss_list

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        self.share_branch_weights()
        LOGGER.info('Shared rgb branch weights with new branch')
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        return self._predict_once(x)

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        x = self.model(x)
        return x

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self.model = self.model._apply(fn)
        return self

    def infer(self, data, is_title):
        out = self.model.infer(data, is_title)
        return out

    def share_branch_weights(self):
        self.model.phase1_e.load_state_dict(deepcopy(self.model.phase1.state_dict()))
        self.model.phase2_e.load_state_dict(deepcopy(self.model.phase2.state_dict()))
        self.model.phase3_e.load_state_dict(deepcopy(self.model.phase3.state_dict()))
        self.model.phase4_e.load_state_dict(deepcopy(self.model.phase4.state_dict()))
        self.model.phase5_e.load_state_dict(deepcopy(self.model.phase5.state_dict()))


class DetNet(nn.Module):
    def __init__(self, nc=80):
        super(DetNet, self).__init__()
        self.nc = nc
        # backbone
        self.phase1 = Conv(c1=3, c2=64, k=3, s=2, p=1)
        self.phase2 = nn.Sequential(
            Conv(c1=64, c2=128, k=3, s=2, p=1),
            C3k2(c1=128, c2=256, n=2, c3k=True, e=0.25)
        )
        self.phase3 = nn.Sequential(
            Conv(c1=256, c2=256, k=3, s=2, p=1),
            C3k2(c1=256, c2=512, n=2, c3k=True, e=0.25)  # l-4
        )
        self.phase4 = nn.Sequential(
            Conv(c1=512, c2=512, k=3, s=2, p=1),
            C3k2(c1=512, c2=512, n=2, c3k=True)  # l-6
        )
        self.phase5 = nn.Sequential(
            Conv(c1=512, c2=512, k=3, s=2, p=1),
            C3k2(c1=512, c2=512, n=2, c3k=True),
            SPPF(c1=512, c2=512, k=5),
            C2PSA(c1=512, c2=512, n=2)  # l-10
        )
        # fpn
        self.up1 = nn.Sequential(
            Upsample(scale_factor=2, mode='nearest'),
            Concat(dimension=1),
            C3k2(c1=1024, c2=512, n=2, c3k=True)  # l-13
        )
        self.up2 = nn.Sequential(
            Upsample(scale_factor=2, mode='nearest'),
            Concat(dimension=1),
            C3k2(c1=1024, c2=256, n=2, c3k=True)  # l-16
        )
        self.down1 = nn.Sequential(
            Conv(c1=256, c2=256, k=3, s=2, p=1),
            Concat(dimension=1),
            C3k2(c1=768, c2=512, n=2, c3k=True)  # l-19
        )
        self.down2 = nn.Sequential(
            Conv(c1=512, c2=512, k=3, s=2, p=1),
            Concat(dimension=1),
            C3k2(c1=1024, c2=512, n=2, c3k=True)  # l-22
        )
        # head
        self.head = Detect(nc=self.nc, ch=(256, 512, 512))

        # evt branch
        self.phase1_e = deepcopy(self.phase1)
        self.phase2_e = deepcopy(self.phase2)
        self.phase3_e = deepcopy(self.phase3)
        self.phase4_e = deepcopy(self.phase4)
        self.phase5_e = deepcopy(self.phase5)

        # temporal module
        self.motion3corr = MotionCorr(in_channels=512)
        self.motion4corr = MotionCorr(in_channels=512)
        self.motion5corr = MotionCorr(in_channels=512)
        self.correlation = SpatialCorrelationSampler(kernel_size=1, patch_size=9, stride=1, padding=0, dilation=1)

    def forward(self, x, *args, **kwargs):
        self.motion3corr.clear_hidden()
        self.motion4corr.clear_hidden()
        self.motion5corr.clear_hidden()
        # first frame
        out_l04 = self.phase3(self.phase2(self.phase1(x[0][0])))  # x[0] is img, just as ix bellow
        self.motion3corr(out_l04, self.correlation)

        out_l06 = self.phase4(out_l04)
        self.motion4corr(out_l06, self.correlation)

        out_l10 = self.phase5(out_l06)
        self.motion5corr(out_l10, self.correlation)

        out_i = self._forward_fpn_head([out_l04, out_l06, out_l10])
        out_list = [out_i]

        # rest frames
        for i, (ix, iy) in enumerate(x[1:]):
            out_l04_e = self.phase3_e(self.phase2_e(self.phase1_e(iy)))
            out_l04_e = self.motion3corr(out_l04_e, self.correlation)

            out_l06_e = self.phase4_e(out_l04_e)
            out_l06_e = self.motion4corr(out_l06_e, self.correlation)

            out_l10_e = self.phase5_e(out_l06_e)
            out_l10_e = self.motion5corr(out_l10_e, self.correlation)

            out_i = self._forward_fpn_head([out_l04_e, out_l06_e, out_l10_e])
            out_list.append(out_i)

        return out_list

    def _forward_fpn_head(self, m_list):
        out_l04, out_l06, out_l10 = m_list

        out_l13 = self.up1[2](self.up1[1](
            [self.up1[0](out_l10), out_l06]
        ))
        out_l16 = self.up2[2](self.up2[1](
            [self.up2[0](out_l13), out_l04]
        ))
        out_l19 = self.down1[2](self.down1[1](
            [self.down1[0](out_l16), out_l13]
        ))
        out_l22 = self.down2[2](self.down2[1](
            [self.down2[0](out_l19), out_l10]
        ))

        out = self.head([out_l16, out_l19, out_l22])

        return out

    def infer(self, sample, is_title=False):
        im = sample['img']
        ev = sample['evt']
        if is_title:
            self.motion3corr.clear_hidden()
            self.motion4corr.clear_hidden()
            self.motion5corr.clear_hidden()

            out3 = self.phase3(self.phase2(self.phase1(im)))
            self.motion3corr(out3, self.correlation)

            out4 = self.phase4(out3)
            self.motion4corr(out4, self.correlation)

            out5 = self.phase5(out4)
            self.motion5corr(out5, self.correlation)

            out = self._forward_fpn_head([out3, out4, out5])
        else:
            out3_e = self.phase3_e(self.phase2_e(self.phase1_e(ev)))
            out3_e = self.motion3corr(out3_e, self.correlation)

            out4_e = self.phase4_e(out3_e)
            out4_e = self.motion4corr(out4_e, self.correlation)

            out5_e = self.phase5_e(out4_e)
            out5_e = self.motion5corr(out5_e, self.correlation)

            out = self._forward_fpn_head([out3_e, out4_e, out5_e])
        return out


    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.head  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self


class MotionCorr(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super(MotionCorr, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.index = torch.tensor([
            0, 2, 4, 6, 8, 10,
            12, 14, 16, 18, 20,
            21, 22, 23, 24, 26, 28, 29, 30,
            31, 32, 33, 34, 36, 38, 39, 40,
            41, 42, 44, 46, 47, 48, 49, 50,
            51, 52, 54, 56, 57, 58, 59, 60,
            62, 64, 66, 68, 70,
            72, 74, 76, 78, 80
        ])

        self.conv_gru = ConvGRUCell(input_dim=in_channels, hidden_dim=hidden_dim, offset_dim=len(self.index))

        self.motion_mask = MotionMask()

        self.hidden_h = None

    def clear_hidden(self):
        self.hidden_h = None

    def forward(self, x, corr):
        if self.hidden_h is None:
            self.hidden_h = x.detach()
            return x
        else:
            b, c, h, w = x.shape

            mo_mask = self.motion_mask(self.hidden_h, x)

            offset_corr = corr(self.hidden_h, x).view(b, -1, h, w) / c
            offset_corr = offset_corr * mo_mask

            new_hidden = self.conv_gru(offset_corr, self.hidden_h)
            self.hidden_h = new_hidden.detach()

            return new_hidden + x


class MotionMask(nn.Module):
    def __init__(self):
        super(MotionMask, self).__init__()
        self.ch_reduce = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1_sattn = self.sigmoid(self.ch_reduce(
            torch.cat([torch.max(x1, dim=1, keepdim=True)[0], torch.mean(x1, dim=1, keepdim=True)], dim=1)
        ))
        x2_sattn = self.sigmoid(self.ch_reduce(
            torch.cat([torch.max(x2, dim=1, keepdim=True)[0], torch.mean(x2, dim=1, keepdim=True)], dim=1)
        ))
        return x1_sattn + x2_sattn


class MotionWarp(nn.Module):
    def __init__(self, in_channels, flow_inch=53, flow_outch=2):
        super(MotionWarp, self).__init__()
        self.in_channels = in_channels
        self.flow_inch = flow_inch
        self.flow_outch = flow_outch
        self.gen_flow = nn.Sequential(
            Conv(c1=flow_inch, c2=32, k=3, s=1, p=1),
            nn.Conv2d(in_channels=32, out_channels=flow_outch, kernel_size=1, stride=1, padding=0),
        )
        self.gen_mask = nn.Sequential(
            Conv(c1=in_channels+flow_inch, c2=128, k=3, s=1, p=1),
            nn.Conv2d(in_channels=128, out_channels=flow_outch+1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, f):
        """
        :param x: feature map
        :param f: flow map
        :return: warped feature map
        """
        f_2c = self.gen_flow(f)  # foreground

        in_x = torch.cat([x, f], dim=1)
        ou_x = self.gen_mask(in_x)

        ou_flow = ou_x[:, :self.flow_outch, :, :]  # background
        ou_mask = ou_x[:, self.flow_outch, :, :]
        ou_mask = torch.sigmoid(torch.unsqueeze(ou_mask, dim=1))

        # offset1 warp * foreground enhance + offset2 warp * background suppress
        flow_up = self._warp_feature(x, f_2c) * (1 - ou_mask) + self._warp_feature(x, ou_flow) * ou_mask

        return x + flow_up

    def _warp_feature(self, x, flow):
        """
        :param x: [n, c, h, w]
        :param flow: [n, 2, h, w]
        :return:
        """
        # get feature size
        B, C, H, W = x.shape
        # get grid coordinate
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], dim=1).to(x)
        # sum flow field and grid coordination, get new coordination
        vgrid = grid + flow
        # normalize to [-1, 1], to match requirement of grid sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :]/max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :]/max(H-1, 1) - 1.0
        # adjust dimension sequence to math requirement of grid sample
        vgrid = vgrid.permute(0, 2, 3, 1)
        # warp by grid sample
        warped_x = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        return warped_x


class ConvGRUCell(nn.Module):
    """
    ConvGRU cell (for single timestep), formula：
        z_t = σ(Conv(x_t, h_{t-1}))
        r_t = σ(Conv(x_t, h_{t-1}))
        h'_t = tanh(Conv(x_t, r_t ⊙ h_{t-1}))
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h'_t
    """

    def __init__(self, input_dim, hidden_dim, offset_dim, kernel_size=3, padding=1):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.offset_dim = offset_dim
        self.kernel_size = kernel_size
        self.padding = padding

        # apply offset to warp apperance map
        self.motion_warp = MotionWarp(in_channels=input_dim, flow_inch=offset_dim)

        # updating gate (z_t)
        self.conv_z = nn.Conv2d(
            in_channels=input_dim + hidden_dim, out_channels=hidden_dim,
            kernel_size=kernel_size, padding=padding
        )
        # reset gate (r_t)
        self.conv_r = nn.Conv2d(
            in_channels=input_dim + hidden_dim, out_channels=hidden_dim,
            kernel_size=kernel_size, padding=padding
        )
        # proposal hidden state (h'_t)
        self.conv_h = nn.Conv2d(
            in_channels=input_dim + hidden_dim, out_channels=hidden_dim,
            kernel_size=kernel_size, padding=padding
        )

    def forward(self, offset, hidden_state=None):
        """
        :param
            x: current input tensor, shape [B, C, H, W]
            hidden_state: precious timestep hidden state, shape [B, hidden_dim, H, W]
        :return
            h_t: new hidden state
        """
        x = self.motion_warp(hidden_state, offset)

        combined = torch.cat([x, hidden_state], dim=1)  # [B, C+hidden_dim, H, W]
        # gate signal
        z_t = torch.sigmoid(self.conv_z(combined))
        r_t = torch.sigmoid(self.conv_r(combined))
        # calculate proposal hidden state
        combined_reset = torch.cat([x, r_t * hidden_state], dim=1)
        h_prime_t = torch.tanh(self.conv_h(combined_reset))
        # update hidden state
        h_t = (1 - z_t) * hidden_state + z_t * h_prime_t

        return h_t


class SpatialCorrelationSampler(nn.Module):
    def __init__(self, kernel_size=1, patch_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch
        assert self.dilation >= 1, "dilation must >=1"
        assert self.dilation_patch >= 1, "dilation_patch must >=1"
        assert self.kernel_size % 2 == 1, "kernel_size must be odd"

        # calculate unfold size
        self._compute_offsets()

        self.offset_index = [
            0, 2, 4, 6, 8, 10,
            12, 14, 16, 18, 20,
            21, 22, 23, 24, 26, 28, 29, 30,
            31, 32, 33, 34, 36, 38, 39, 40,
            41, 42, 44, 46, 47, 48, 49, 50,
            51, 52, 54, 56, 57, 58, 59, 60,
            62, 64, 66, 68, 70,
            72, 74, 76, 78, 80
        ]

    def _compute_offsets(self):
        """calculate offsets in neighbors"""
        h_radius = self.patch_size[0] // 2
        w_radius = self.patch_size[1] // 2

        offsets = []
        for i in range(-h_radius, h_radius + 1):
            for j in range(-w_radius, w_radius + 1):
                offsets.append((
                    i * self.dilation_patch,
                    j * self.dilation_patch
                ))
        self.offsets = offsets
        self.num_offsets = len(offsets)

    def _get_padding(self):
        """calculate padding size"""
        pad_h = self.padding + (self.patch_size[0] // 2) * self.dilation_patch
        pad_w = self.padding + (self.patch_size[1] // 2) * self.dilation_patch
        return (pad_h, pad_w)

    def forward(self, input1, input2):
        """
        :param input1: [b, c, h, w]
        :param input2: [b, c, h, w]
        :return: [b, patch_w*patch_h, h_out, w_out]
        """
        assert input1.shape == input2.shape
        B, C, H, W = input1.shape

        # get output size
        pad_h, pad_w = self._get_padding()
        H_out = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # pad input2
        input2_pad = F.pad(input2, [pad_w, pad_w, pad_h, pad_h])

        # unfold input1 (with dilation)
        input1_unfold = F.unfold(
            input1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )  # [B, C*kh*kw, L]
        # L = input1_unfold.size(2)

        # initialize output
        output = []
        # for h_off, w_off in self.offsets:
        for ofs_idx in self.offset_index:
            h_off, w_off = self.offsets[ofs_idx]
            # get sampling region in input2
            h_start = pad_h + h_off
            w_start = pad_w + w_off
            h_end = h_start + (H + 2 * self.padding) * self.dilation
            w_end = w_start + (W + 2 * self.padding) * self.dilation

            # extract corresponding region
            input2_slice = input2_pad[:, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation]

            # unfold input2_slice
            input2_unfold = F.unfold(
                input2_slice,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0,
                dilation=self.dilation
            )  # [B, C*kh*kw, L]

            # calculate correlation
            corr = (input1_unfold * input2_unfold).sum(dim=1)  # [B, L]
            corr = corr.view(B, 1, H_out, W_out)
            output.append(corr)

        # cat all position
        out = torch.cat(output, dim=1)
        return out
