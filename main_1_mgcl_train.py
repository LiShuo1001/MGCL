import torch
import argparse
from alisuretool.Tools import Tools
from net.net_tools import MGCLNetwork
from dataset.dataset_tools import FSSDataset, Evaluator
from util.util_tools import MyCommon, MyOptim, AverageMeter, Logger


class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=True, gpu_id=args.gpuid)

        self.model = MGCLNetwork(args).to(self.device)
        self.optimizer = MyOptim.get_finetune_optimizer(args, self.model)

        FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
        self.dataloader_train = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, 8, args.fold, 'train', use_mask=args.mask, mask_num=args.mask_num)
        self.dataloader_val = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, 8, args.fold, 'val', use_mask=args.mask, mask_num=args.mask_num)

        Logger.log_params(self.model)
        pass

    def train(self):
        best_val_miou = 0
        for epoch in range(self.args.epoch_num):
            Tools.print("begin epoch {} train".format(epoch))

            now_lr = MyOptim.adjust_learning_rate_poly(self.args, self.optimizer, epoch, self.args.epoch_num)
            Tools.print("lr={}".format(now_lr))
            train_loss, train_miou, train_fb_iou = self.train_one_epoch(epoch, self.dataloader_train)

            Tools.print("begin epoch {} test".format(epoch))
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = self.test_one_epoch(epoch, self.dataloader_val)
                pass

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(self.model, epoch, val_miou)
                pass

            Logger.tbd_writer.add_scalars('data/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'train_miou': train_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'train_fb_iou': train_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
            pass
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
        pass

    def train_one_epoch(self, epoch, dataloader):
        has_module = hasattr(self.model, "module")
        self.model.module.train_mode() if has_module else self.model.train_mode()
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            logit = self.model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_labels'].squeeze(1),
                               query_mask=batch['query_mask'] if 'query_mask' in batch else None,
                               support_masks=batch['support_masks'].squeeze(1) if 'support_masks' in batch else None)
            # 2. Compute loss & update model parameters
            if has_module:
                loss = self.model.module.compute_objective(logit, batch['query_label'])
            else:
                loss = self.model.compute_objective(logit, batch['query_label'])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 3. Evaluate prediction
            pred = logit.argmax(dim=1)
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
            pass

        # Write evaluation results
        average_meter.write_result('Training', epoch)
        avg_loss = MyCommon.mean(average_meter.loss_buf)
        miou, fb_iou = average_meter.compute_iou()
        return avg_loss, miou, fb_iou

    def test_one_epoch(self, epoch, dataloader):
        has_module = hasattr(self.model, "module")
        self.model.module.eval() if has_module else self.model.eval()
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            logit = self.model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_labels'].squeeze(1),
                               query_mask=batch['query_mask'] if 'query_mask' in batch else None,
                               support_masks=batch['support_masks'].squeeze(1) if 'support_masks' in batch else None)
            # 2. Compute loss & update model parameters
            if has_module:
                loss = self.model.module.compute_objective(logit, batch['query_label'])
            else:
                loss = self.model.compute_objective(logit, batch['query_label'])

            # 3. Evaluate prediction
            pred = logit.argmax(dim=1)
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
            pass

        # Write evaluation results
        average_meter.write_result('Validation', epoch)
        avg_loss = MyCommon.mean(average_meter.loss_buf)
        miou, fb_iou = average_meter.compute_iou()
        return avg_loss, miou, fb_iou

    pass


def my_parser_isaid():
    # Arguments parsing
    parser = argparse.ArgumentParser(description='MGCL Pytorch Implementation')

    parser.add_argument('--logpath', type=str, default='demo')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument("--finetune_backbone", type=bool, default=True)
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--mask_num', type=int, default=128)

    parser.add_argument('--datapath', type=str,
                        default='/8T/data/ubuntu1080/FSS-RS/remote_sensing/iSAID_patches')
    parser.add_argument('--benchmark', type=str, default='isaid', choices=['isaid', 'dlrsd'])
    parser.add_argument('--is-sgd', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()
    Logger.initialize(args, training=True)
    return args


def my_parser_dlrsd():
    # Arguments parsing
    parser = argparse.ArgumentParser(description='MGCL Pytorch Implementation')

    parser.add_argument('--logpath', type=str, default='demo')
    parser.add_argument('--gpuid', type=int, default=1)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument("--finetune_backbone", type=bool, default=True)
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--mask_num', type=int, default=128)

    parser.add_argument('--datapath', type=str, default='/8T/data/ubuntu1080/FSS-RS/DLRSD')
    parser.add_argument('--benchmark', type=str, default='dlrsd', choices=['isaid', 'dlrsd'])
    parser.add_argument('--is-sgd', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()
    Logger.initialize(args, training=True)
    return args


if __name__ == '__main__':
    runner = Runner(args=my_parser_isaid())
    # runner = Runner(args=my_parser_dlrsd())
    runner.train()
    pass
