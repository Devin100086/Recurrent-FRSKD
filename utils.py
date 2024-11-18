import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


def create_logging(path_log):
    logger = logging.getLogger('Result_log')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path_log)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    return logger

def train_without_distill(model, optimizer, criterion, train_loader, args):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion_ce, _ = criterion
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        feats, outputs = model(inputs)
        loss = criterion_ce(outputs, targets)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_size = targets.size(0)
    losses.update(loss.item(), batch_size)
    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    top1.update(acc1, batch_size)

    return losses.avg, top1.avg

def train(model, bifpn, optimizer, criterion, train_loader, args):
    model.train()
    bifpn.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion_ce, criterion_kd = criterion
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        if args.aux == 'sla':
            outputs, bi_outputs, feats, bi_feats, loss = criterion_ce(model, bifpn, inputs, targets)
            loss += criterion_kd(outputs, bi_outputs, feats, bi_feats)
            outputs = outputs[:targets.size(0)]

        else:
            if args.aug == 'mixup':
                mixed_inputs, targets_1, targets_2, lam = mixup_data(inputs, targets, args.aug_a)

                feats, outputs = model(mixed_inputs, args.distill == 'od')
                feats = feats[-args.num_features:]
                bi_feats, bi_outputs = bifpn(feats, args.distill == 'od')

                loss_model = mixed_criterion(outputs, targets_1, targets_2, lam)
                loss_bifpn = mixed_criterion(bi_outputs, targets_1, targets_2, lam)
                loss_model += criterion_kd(outputs, bi_outputs, feats, bi_feats)
                loss = loss_model + loss_bifpn

            elif args.aug == 'cutmix':
                mixed_inputs, targets_1, targets_2, lam = cutmix_data(inputs, targets, args.aug_a)

                feats, outputs = model(mixed_inputs, args.distill == 'od')
                feats = feats[-args.num_features:]
                bi_feats, bi_outputs = bifpn(feats, args.distill == 'od')

                loss_model = mixed_criterion(outputs, targets_1, targets_2, lam)
                loss_bifpn = mixed_criterion(bi_outputs, targets_1, targets_2, lam)
                loss_model += criterion_kd(outputs, bi_outputs, feats, bi_feats)
                loss = loss_model + loss_bifpn

            else:
                feats, outputs = model(inputs, args.distill == 'od')
                feats = feats[-args.num_features:]
                bi_feats, bi_outputs = bifpn(feats, args.distill == 'od')

                loss_model = criterion_ce(outputs, targets)
                loss_bifpn = criterion_ce(bi_outputs, targets)
                loss_model += criterion_kd(outputs, bi_outputs, feats, bi_feats)
                loss = loss_model + loss_bifpn
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    return losses.avg, top1.avg, top5.avg


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_size = targets.size(0)
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                feats, outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


def lr_scheduler(optimizer, scheduler, schedule, lr_decay, total_epoch):
    optimizer.zero_grad()
    optimizer.step()
    if scheduler == 'step':
        return optim.lr_scheduler.MultiStepLR(optimizer, schedule, gamma=lr_decay)
    elif scheduler == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
    else:
        raise NotImplementedError('{} learning rate is not implemented.')


def mixed_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).cuda()
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha):
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


import matplotlib.pyplot as plt
import numpy as np
import cv2
def generate_attention_map(feature_maps):
    # 使用全局平均池化
    def channel_wise_pooling(feature_map):
        attention_map = F.adaptive_avg_pool2d(feature_map, (1, 1))
        return attention_map
    # 将注意图转换为与输入图像相同的大小
    def upsample_attention_map(attention_map, size):
        return F.interpolate(attention_map, size=size, mode='bilinear', align_corners=False)
    # 归一化注意力图
    def normalize_attention_map(attention_map):
        attention_map = attention_map.squeeze()
        attention_map -= attention_map.min()
        attention_map /= attention_map.max()
        return attention_map
    # 将注意图转换为彩色
    def apply_colormap(attention_map):
        attention_map = attention_map.cpu().numpy()
        attention_map = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        return attention_map
    attention_maps = [channel_wise_pooling(feature_map) for feature_map in feature_maps]

# 将注意图叠加到原始图像上
def overlay_attention_map(input_image, attention_map):
    attention_map = cv2.cvtColor(attention_map, cv2.COLOR_BGR2RGB)
    input_image = np.array(input_image.resize((224, 224)))
    overlayed_image = cv2.addWeighted(input_image, 0.6, attention_map, 0.4, 0)
    return overlayed_image

