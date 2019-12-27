"""Train CenterNet"""
import os
import sys
dlcv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, dlcv_path)

import argparse
import dlcv
from dlcv.model_zoo import get_model
import torch
import torchvision
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensor

import ignite
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.utils import convert_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Train CenterNet networks")
    parser.add_argument('--network', type=str, default='resnet18_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--dataset-root', type=str, default='~/.mxnet/datasets/',
                        help='Path of the directory where the dataset is located.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=140,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=1.25e-4,
                        help='Learning rate, default is 0.000125')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='90,120',
                        help='epochs at which learning rate decays. default is 90,120.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='Weight decay, default is 1e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--wh-weight', type=float, default=0.1,
                        help='Loss weight for width/height')
    parser.add_argument('--center-reg-weight', type=float, default=1.0,
                        help='Center regression loss weight')
    parser.add_argument('--flip-validation', action='store_true',
                        help='flip data augmentation in validation.')

    args = parser.parse_args()
    return args

def get_transform(aug):
    def transform(data, target):
        transform = A.Compose(aug, A.BboxParams(format='pascal_voc', label_fields=['catid']))
        augmented = transform(image=data, bboxes=target[:, :4], catid=target[:, 4])
        bboxes = [(*box, label) for box, label in zip(augmented['bboxes'], augmented['catid'])]
        return augmented['image'], bboxes 
    return transform

def get_dataset(dataset, args):
    width = height = args.data_shape
    train_aug = [
        A.HorizontalFlip(),
        A.RandomSizedBBoxSafeCrop(width, height),
        A.RGBShift(),
        A.Blur(blur_limit=11),
        A.RandomBrightness(),
        A.CLAHE(),
        A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ToTensor(),
    ]

    val_aug = [
        A.Resize(width, height),
        A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ToTensor(),
    ]

    if dataset.lower() == 'voc':
        train_dataset = dlcv.data.VOCDetection(
            root='/home/vismarty/.datasets/VOCdevkit',
            splits=[(2007, 'trainval'), (2012, 'trainval')],
            transform=get_transform(train_aug)
            )
        val_dataset = dlcv.data.VOCDetection(
            root='/home/vismarty/.datasets/VOCdevkit',
            splits=[(2007, 'test')],
            transform=get_transform(val_aug)
            )
        # val_metric = dlcv.utils.metrics.voc_detection.VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        val_metric = None
    # elif dataset.lower() == 'coco':
    #     train_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_train2017')
    #     val_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_val2017', skip_empty=False)
    #     val_metric = COCODetectionMetric(
    #         val_dataset, args.save_prefix + '_eval', cleanup=True,
    #         data_shape=(args.data_shape, args.data_shape), post_affine=get_post_transform)
    #     # coco validation is slow, consider increase the validation interval
    #     if args.val_interval == 1:
    #         args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    return train_dataset, val_dataset, val_metric



def get_dataloader(net, train_dataset, val_dataset, batch_size, input_size, num_workers, device):
    """Get dataloader."""
    num_class = len(train_dataset.classes)
    pin_memory = False if device == torch.device('cpu') else True
    fake_x = torch.rand((1, 3, input_size, input_size)).to(device)
    net.eval()
    with torch.no_grad():
        results = net.base_network(fake_x)
    input_height, input_width = results.size(2), results.size(3)

    collate_fn = dlcv.model_zoo.CenterNetTargetGenerator(num_class, input_size, input_size,
                                                         input_height, input_width)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, 
                              pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=collate_fn)
    return train_loader, val_loader
    
def train(net, train_data, val_data, eval_metric, device, args):
    """Training pipeline"""
    net.to(device)

    heatmap_loss = dlcv.loss.HeatmapFocalLoss(from_logits=True)
    wh_loss = dlcv.loss.MaskedL1Loss(weight=args.wh_weight)
    center_reg_loss = dlcv.loss.MaskedL1Loss(weight=args.center_reg_weight)
    optimizer = torch.optim.Adam(net.parameters())


    def process_function(engine, batch):
        data, heatmap_targets, wh_targets, wh_masks, center_reg_targets, center_reg_masks = \
            [convert_tensor(s, device) for s in batch]
        net.train()
        optimizer.zero_grad()
        heatmap_pred, wh_pred, center_reg_pred = net(data)
        wh_loss_ = wh_loss(wh_pred, wh_targets, wh_masks)
        center_reg_loss_ = center_reg_loss(center_reg_pred, center_reg_targets, center_reg_masks)
        heatmap_loss_ = heatmap_loss(heatmap_pred, heatmap_targets)
        total_loss_ = wh_loss_ + center_reg_loss_ + heatmap_loss_
        total_loss_.backward()
        optimizer.step()
        return total_loss_.item(), heatmap_loss_.item(), center_reg_loss_.item(), wh_loss_.item()
        

    trainer = Engine(process_function)

    log_interval = 10
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter_ = (engine.state.iteration - 1) % len(train_data) + 1
        if iter_ % log_interval == 0:
            print("Epoch[{}] step[{}/{}] total_loss: {:.3f}, heatmap_loss: {:.3f}, "
                  "center_loss: {:.3f}, wh_loss: {:.3f}".format(
                engine.state.epoch, iter_, len(train_data),  *engine.state.output))

    trainer.run(train_data, max_epochs=10)



if __name__ == '__main__':
    args = parse_args()

    # fix seed for torch, numpy and python builtin randm generator.
    dlcv.utils.random.seed(args.seed)

    # training devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # network
    net_name = '_'.join(('center_net', args.network, args.dataset))
    net = get_model(net_name, pretrained_base=True, norm_layer=torch.nn.BatchNorm2d, device=device)
    # net.eval()
    net.train()
    # fake_x = torch.rand((1, 3, 512, 512)).to(device)
    # with torch.no_grad():
    #     results = net(fake_x)
    # print(results[0].size())

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    batch_size = args.batch_size
    train_data, val_data = get_dataloader(net, train_dataset, val_dataset, batch_size, args.data_shape, args.num_workers, device)

    # train_iter = iter(train_data)
    # sample = next(train_iter)

    train(net, train_data, val_data, eval_metric, device, args)