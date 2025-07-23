import torch
import argparse
from alisuretool.Tools import Tools
from net.net_tools import MGCLNetwork
from util.util_tools import MyCommon, AverageMeter, Logger
from dataset.dataset_tools import FSSDataset, Evaluator


class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=self.args.use_gpu, gpu_id=args.gpuid)

        self.model = MGCLNetwork(args).to(self.device)
        self.model.eval()
        weights = torch.load(args.load, map_location=None if self.args.use_gpu else torch.device('cpu'))
        weights = {one.replace("module.", ""): weights[one] for one in weights.keys()}
        weights = {one.replace("hpn_learner.", "mgcd."): weights[one] for one in weights.keys()}
        self.model.load_state_dict(weights)

        FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
        self.dataloader_val = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.shot,
            use_mask=args.mask, mask_num=args.mask_num)
        pass

    @torch.no_grad()
    def test(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            pred = self.model.predict_nshot(batch)

            # 2. Evaluate prediction
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
            pass
        miou, fb_iou = average_meter.compute_iou()
        return miou, fb_iou

    @torch.no_grad()
    def test_class(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            pred = self.model.predict_nshot(batch)

            # 2. Evaluate prediction
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
            pass
        miou, fb_iou, iou = average_meter.compute_iou_class()
        return miou, fb_iou, iou

    pass


def my_parser(fold=0, shot=1, backbone='resnet50', benchmark="isaid",
              load='./logs/demo/best_model.pt', use_gpu=False, gpu_id=0,
              bsz=2, mask=True, mask_num=128):
    datapath = None
    if benchmark == "isaid":
        datapath = '/mnt/4T/ALISURE/FSS-RS/remote_sensing/iSAID_patches'
    elif benchmark == "dlrsd":
        datapath = '/mnt/4T/ALISURE/FSS-RS/DLRSD'

    parser = argparse.ArgumentParser(description='MGCL Pytorch Implementation')

    parser.add_argument('--logpath', type=str, default='demo')
    parser.add_argument('--use_gpu', type=bool, default=use_gpu)
    parser.add_argument('--gpuid', type=int, default=gpu_id)
    parser.add_argument('--bsz', type=int, default=bsz)
    parser.add_argument('--fold', type=int, default=fold, choices=[0, 1, 2])
    parser.add_argument('--shot', type=int, default=shot, choices=[1, 5])
    parser.add_argument('--load', type=str, default=load)
    parser.add_argument('--backbone', type=str, default=backbone,
                        choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--mask', type=bool, default=mask)
    parser.add_argument('--mask_num', type=int, default=mask_num)
    parser.add_argument('--datapath', type=str, default=datapath)
    parser.add_argument('--benchmark', type=str, default=benchmark, choices=['isaid', 'dlrsd'])
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    MyCommon.fix_randseed(0)

    # isaid resnet50
    # 2023-10-24 09:54:36 fold=0, shot=1, mIoU: 42.77439880371094 FB-IoU: 62.60266876220703
    # 2023-11-03 14:01:02 iou: [0.3375, 0.4660, 0.4952, 0.5550, 0.2850]
    # 2023-10-24 10:04:32 fold=0, shot=5, mIoU: 49.13707733154297 FB-IoU: 66.159912109375
    args = my_parser(fold=0, shot=1, backbone='resnet50', benchmark="isaid",
                     use_gpu=True, gpu_id=0, bsz=8, mask=True, mask_num=128,
                     load='./logs/resnet50_fold0/best_model.pt')
    # 2023-10-24 09:57:40 fold=1, shot=1, mIoU: 30.592941284179688 FB-IoU: 55.60072708129883
    # 2023-11-03 14:01:54 iou: [0.3214, 0.4551, 0.3579, 0.1319, 0.2634]
    # 2023-10-24 10:06:42 fold=1, shot=5, mIoU: 32.57774353027344 FB-IoU: 57.06192398071289
    # args = my_parser(fold=1, shot=1, backbone='resnet50', benchmark="isaid",
    #                  use_gpu=True, gpu_id=0, bsz=8, mask=True, mask_num=128,
    #                  load='./logs/resnet50_fold1/best_model.pt')

    # 2023-10-24 09:57:02 fold=2, shot=1, mIoU: 46.38703918457031 FB-IoU: 58.5185661315918
    # 2023-11-03 14:02:02 iou: [0.4573, 0.6166, 0.4611, 0.5389, 0.2456],
    # 2023-10-24 10:25:51 fold=2, shot=5, mIoU: 52.74596405029297 FB-IoU: 62.11357116699219
    # args = my_parser(fold=2, shot=1, backbone='resnet50', benchmark="isaid",
    #                  use_gpu=True, gpu_id=0, bsz=8, mask=True, mask_num=128,
    #                  load='./logs/resnet50_fold2/best_model.pt')

    Logger.initialize(args, training=False)
    runner = Runner(args=args)
    Logger.log_params(runner.model)

    # miou, fb_iou = runner.test()
    # Tools.print("mIoU: {} FB-IoU: {}".format(miou, fb_iou))
    miou, fb_iou, iou = runner.test_class()
    Tools.print("mIoU: {} FB-IoU: {} iou: {}".format(miou, fb_iou, iou))
    pass

