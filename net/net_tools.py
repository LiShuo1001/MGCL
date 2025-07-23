import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg


class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y

    pass


class MGCDModule(nn.Module):

    def __init__(self, inch):
        super().__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3, outch4 = 16, 32, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv2d(outch4, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder2 = nn.Sequential(
            nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))
        pass

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid, query_mask=None):
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        if query_mask is not None:
            # MGFE
            _hypercorr_encoded = MGFEModule.update_feature_one(hypercorr_encoded, query_mask)
            hypercorr_encoded = torch.concat([hypercorr_encoded, _hypercorr_encoded], dim=1)
        else:
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
            pass

        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size,
                                          mode='bilinear', align_corners=True)
        logit = self.decoder2(hypercorr_decoded)
        return logit

    pass


class SegmentationHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        # MGFE
        _query_feats, _support_feats = MGFEModule.update_feature(
            query_feats, support_feats, query_mask, support_masks)
        query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
        support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]

        # FBC
        support_feats_fg = [self.label_feature(
            support_feat, support_label.clone())for support_feat in support_feats]
        support_feats_bg = [self.label_feature(
            support_feat, (1 - support_label).clone())for support_feat in support_feats]
        corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
        corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
        corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                  dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

        # MGCD
        logit = self.mgcd(corr[::-1], query_mask)
        return logit

    @staticmethod
    def label_feature(feature, label):
        label = F.interpolate(label.unsqueeze(1).float(), feature.size()[2:],
                             mode='bilinear', align_corners=True)
        return feature * label

    @staticmethod
    def multilayer_correlation(query_feats, support_feats):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)
            pass

        return corrs

    pass


class MGFEModule(object):

    @classmethod
    def update_feature_one(cls, query_feat, query_mask):
        return cls.enabled_feature([query_feat], query_mask)[0]

    @classmethod
    def update_feature(cls, query_feats, support_feats, query_mask, support_masks):
        query_feats = cls.enabled_feature(query_feats, query_mask)
        support_feats = cls.enabled_feature(support_feats, support_masks)
        return query_feats, support_feats

    @classmethod
    def enabled_feature(cls, feats, masks):
        b, m, w, h = masks.shape
        index_mask = torch.zeros_like(masks[:, 0]).long() + m
        for i in range(m):
            index_mask[masks[:, i]==1] = i
        masks = torch.nn.functional.one_hot(index_mask)[:, :, :, :m].permute((0, 3, 1, 2))

        enabled_feats = []
        for feat in feats:
            target_masks = F.interpolate(masks.float(), feat.shape[-2:], mode='nearest')
            map_features = cls.my_masked_average_pooling(feat, target_masks)

            b, m, w, h = target_masks.shape
            _, _, c = map_features.shape
            _map_features = map_features.permute(0, 2, 1).contiguous()
            feature_sum = _map_features @ target_masks.view(b, m, -1)
            feature_sum = feature_sum.view(b, c, w, h)

            sum_mask = target_masks.sum(dim=1, keepdim=True)
            enabled_feat = torch.div(feature_sum, sum_mask + 1e-8)
            enabled_feats.append(enabled_feat)
            pass
        return enabled_feats

    @staticmethod
    def my_masked_average_pooling(feature, mask):
        b, c, w, h = feature.shape
        _, m, _, _ = mask.shape

        _mask = mask.view(b, m, -1)
        _feature = feature.view(b, c, -1).permute(0, 2, 1).contiguous()
        feature_sum = _mask @ _feature
        masked_sum = torch.sum(_mask, dim=2, keepdim=True)

        masked_average_pooling = torch.div(feature_sum, masked_sum + 1e-8)
        return masked_average_pooling

    pass


class MGCLNetwork(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone_type = args.backbone
        self.finetune_backbone = args.finetune_backbone if hasattr(args, "finetune_backbone") else False

        if "vgg" in self.backbone_type:
            self.backbone = vgg.vgg16(pretrained=True)
            self.extract_feats = self.extract_feats_vgg
        elif "50" in self.backbone_type:
            self.backbone = resnet.resnet50(pretrained=True)
            self.extract_feats = self.extract_feats_res
        else:
            self.backbone = resnet.resnet101(pretrained=True)
            self.extract_feats = self.extract_feats_res
            pass

        self.segmentation_head = SegmentationHead()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        pass

    def forward(self, query_img, support_img, support_label, query_mask, support_masks):
        if self.finetune_backbone:
            query_feats = self.extract_feats(query_img, self.backbone)
            support_feats = self.extract_feats(support_img, self.backbone)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img, self.backbone)
                support_feats = self.extract_feats(support_img, self.backbone)
                pass
            pass
        # MGFE, FBC, MGCD
        logit = self.segmentation_head(query_feats, support_feats, support_label.clone(), query_mask, support_masks)
        logit = F.interpolate(logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        return logit

    def predict_nshot(self, batch):
        nshot = batch["support_imgs"].shape[1]
        logit_label_agg = 0
        for s_idx in range(nshot):
            logit_label = self.forward(
                batch['query_img'], batch['support_imgs'][:, s_idx],  batch['support_labels'][:, s_idx],
                query_mask=batch['query_mask'] if 'query_mask' in batch and self.args.mask else None,
                support_masks=batch['support_masks'][:, s_idx] if 'support_masks' in batch and self.args.mask else None)

            result_i = logit_label.argmax(dim=1).clone()
            logit_label_agg += result_i

            # One-Shot
            if nshot == 1: return result_i.float()
            pass

        # Few-Shot
        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_label_agg.size(0)
        max_vote = logit_label_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_label = logit_label_agg.float() / max_vote
        threshold = 0.4
        pred_label[pred_label < threshold] = 0
        pred_label[pred_label >= threshold] = 1
        return pred_label

    def compute_objective(self, logit_label, gt_label):
        bsz = logit_label.size(0)
        logit_label = logit_label.view(bsz, 2, -1)
        gt_label = gt_label.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_label, gt_label)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
        pass

    @staticmethod
    def extract_feats_vgg(img, backbone):
        feat_ids = [16, 23, 30]
        feats = []
        feat = img
        for lid, module in enumerate(backbone.features):
            feat = module(feat)
            if lid in feat_ids:
                feats.append(feat.clone())
        return feats

    @staticmethod
    def extract_feats_res(img, backbone):
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(img))))

        feats = []
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        feats.append(x.clone())
        x = backbone.layer3(x)
        feats.append(x.clone())
        x = backbone.layer4(x)
        feats.append(x.clone())
        return feats

    pass

