import os
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
import PIL.Image as Image
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import vgg
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
from torchvision.models import resnet
from torch.utils.data import DataLoader
from net.net_tools import MGCLNetwork
from util.util_tools import MyCommon, MyOptim, AverageMeter, Logger
from dataset.dataset_tools import FSSDataset, Evaluator, DatasetISAID


class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=True, gpu_id=args.gpuid)

        self.model = MGCLNetwork(args, False).to(self.device)
        if self.args.backbone == "resnet101":
            self.model = nn.DataParallel(self.model)
            pass

        self.optimizer = MyOptim.get_finetune_optimizer(args, self.model)

        FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath,
                              use_original_imgsize=args.use_original_imgsize)
        self.dataloader_train = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, args.nworker, args.fold, 'train',
            use_mask=args.mask, mask_num=args.mask_num)
        self.dataloader_val = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, args.nworker, args.fold, 'val',
            use_mask=args.mask, mask_num=args.mask_num)

        Logger.log_params(self.model)
        pass

    def train(self):
        best_val_miou = 0
        for epoch in range(self.args.niter):
            Tools.print("begin epoch {} train".format(epoch))
            train_loss, train_miou, train_fb_iou = self.train_one_epoch(
                epoch, self.dataloader_train, training=True)
            Tools.print("begin epoch {} test".format(epoch))
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = self.train_one_epoch(
                    epoch, self.dataloader_val, training=False)

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(self.model, epoch, val_miou)

            Logger.tbd_writer.add_scalars('data/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'train_miou': train_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'train_fb_iou': train_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
            pass
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
        pass

    def train_one_epoch(self, epoch, dataloader, training):
        # Force randomness during training / freeze randomness during testing
        MyCommon.fix_randseed(None) if training else MyCommon.fix_randseed(0)
        if hasattr(self.model, "module"):
            self.model.module.train_mode() if training else self.model.module.eval()
        else:
            self.model.train_mode() if training else self.model.eval()
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        now_lr = 0
        for idx, batch in enumerate(dataloader):
            if training:
                now_lr = MyOptim.adjust_learning_rate_poly(
                    self.args, self.optimizer, epoch * len(dataloader))
                pass

            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            logit = self.model(
                batch['query_img'], batch['support_imgs'].squeeze(1),
                batch['support_labels'].squeeze(1),
                query_mask=batch['query_mask'] if 'query_mask' in batch else None,
                support_masks=batch['support_masks'].squeeze(1) if 'support_masks' in batch else None)
            # 2. Compute loss & update model parameters
            if hasattr(self.model, "module"):
                loss = self.model.module.compute_objective(logit, batch['query_label'])
            else:
                loss = self.model.compute_objective(logit, batch['query_label'])
            loss = loss
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pass

            # 3. Evaluate prediction
            pred = logit.argmax(dim=1)
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
            pass
        Tools.print("lr={}".format(now_lr))

        # Write evaluation results
        average_meter.write_result('Training' if training else 'Validation', epoch)
        avg_loss = MyCommon.mean(average_meter.loss_buf)
        miou, fb_iou = average_meter.compute_iou()
        return avg_loss, miou, fb_iou

    pass


def my_parser():
    # Arguments parsing
    parser = argparse.ArgumentParser(description='MGCL Pytorch Implementation')

    parser.add_argument('--logpath', type=str, default='demo')
    parser.add_argument('--gpuid', type=int, default=1)
    parser.add_argument('--bsz', type=int, default=15)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument("--finetune_backbone", type=bool, default=True)
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--mask_num', type=int, default=128)

    parser.add_argument('--datapath', type=str,
                        default='/mnt/4T/ALISURE/FSS-RS/remote_sensing/iSAID_patches')
    parser.add_argument('--benchmark', type=str, default='isaid', choices=['isaid', 'dlrsd'])
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--use_original_imgsize', type=bool, default=False)
    args = parser.parse_args()
    Logger.initialize(args, training=True)
    return args


"""
Fold 0
Backbone # param.: 23561205
Learnable # param.: 700818
Total # param.: 24262023
*** Validation [@Epoch 11] Avg L: 0.33205  mIoU: 42.77   FB-IoU: 62.60   ***
Model saved @11 w/ val. mIoU: 42.77.

Fold 1
Backbone # param.: 23561205
Learnable # param.: 700818
Total # param.: 24262023
*** Validation [@Epoch 20] Avg L: 0.32656  mIoU: 30.59   FB-IoU: 55.60   ***
Model saved @20 w/ val. mIoU: 30.59.

Fold 2
Backbone # param.: 23561205
Learnable # param.: 700818
Total # param.: 24262023
*** Validation [@Epoch 33] Avg L: 0.41115  mIoU: 46.39   FB-IoU: 58.52   ***
Model saved @33 w/ val. mIoU: 46.39.
"""


if __name__ == '__main__':
    runner = Runner(args=my_parser())
    runner.train()
    pass
