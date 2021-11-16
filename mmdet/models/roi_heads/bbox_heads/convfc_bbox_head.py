import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 proto=0,
                 ranking=0,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.proto = proto
        self.ranking = ranking
        if proto:
            self.proto_mip = self._proto_mapping(2048, fc_out_channels)

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def _proto_mapping(self, in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels)
        )
        return mlp

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.proto:
            if self.proto == 1:
                prototype = torch.load(
                    '/prototype/prototype.pth') #You have to change the base dir.
            if self.proto == 2:
                prototype = torch.load(
                    '/prototype/prototype_val.pth') #You have to change the base dir.
            bg_proto = prototype.mean(dim=0).unsqueeze(0)
            prototype = torch.cat((prototype, bg_proto), 0).cuda()
            proto_w = self.proto_mlp(prototype).requires_grad_()
            cls_score = torch.mm(x_cls, proto_w.T) if self.with_cls else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None
        if self.training:
            if self.ranking:
                _, idx = cls_score.topk(1)
                for i in range(len(cls_score)):
                    if cls_score.shape[1] - 1 not in idx[i]:
                        if self.ranking == 1:
                            cls_score[i][-1] = (cls_score[i][idx[i][0]] + cls_score[i][idx[i][1]]) / 2
                        elif 0 < self.ranking < 1:
                            cls_score[i][-1] = cls_score[i][idx[i][0]] * self.ranking
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
