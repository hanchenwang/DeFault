import datetime
import os
import sys
import time
import torch
from torch import nn
import torchvision
from torchvision.transforms import Compose
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import utils
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T
from functools import reduce
import operator
import mlreal_functions as mlr
import random


step = 0

def train_one_epoch(model, criterion, optimizer, lr_scheduler, 
                    dataloader, device, epoch, print_freq, writer):
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for data, label in metric_logger.log_every(dataloader, print_freq, header):

        start_time = time.time()
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        output = model(data)

#        batch_size = data.shape[0]

#        rand_file = random.randint(0, 6)
#        rand_data = random.randint(0, 31-batch_size)
#        starting_id = (rand_file*500+
#        match_data = field_data_all[rand_data:rand_data+batch_size,:,:,:].view(batch_size,1,1000,31)
#        match_data.to(device)
#        match_data_conv, input_data_conv = mlr.get_mlreal(match_data,data)
#        match_data_conv = T.minmax_normalize(match_data_conv,-7.0,7.0)
#        input_data_conv = T.minmax_normalize(input_data_conv,-7.0,7.0)

#        output = model(input_data_conv)

        loss, loss_g1v, loss_g2v = criterion(output, label)

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_g1v_val = loss_g1v.item()
        loss_g2v_val = loss_g2v.item()
        batch_size = data.shape[0]
        metric_logger.update(loss=loss_val, loss_g1v=loss_g1v_val, 
            loss_g2v=loss_g2v_val, lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        if writer:
            writer.add_scalar('loss', loss_val, step)
            writer.add_scalar('loss_g1v', loss_g1v_val, step)
            writer.add_scalar('loss_g2v', loss_g2v_val, step)
        step += 1


def evaluate(model, criterion, dataloader, device, writer):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    with torch.no_grad():
        for data, label in metric_logger.log_every(dataloader, 20, header):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data)
#            batch_size = data.shape[0]

#            rand_data = 0#random.randint(0, 31-batch_size)

#            match_data = field_data_all[rand_data:rand_data+batch_size,:,:,:].view(batch_size,1,1000,31)
#            match_data.to(device)
#            match_data_conv, input_data_conv = mlr.get_mlreal(match_data,data)
#            match_data_conv = T.minmax_normalize(match_data_conv,-7.0,7.0)
#            input_data_conv = T.minmax_normalize(input_data_conv,-7.0,7.0)

#            output = model(input_data_conv)
#            print('label:',label.shape,label)
#            print('pred:',output.shape,output)
            loss, loss_g1v, loss_g2v = criterion(output, label)
            batch_size = data.shape[0]
            metric_logger.update(loss=loss.item(), 
                loss_g1v=loss_g1v.item(), 
                loss_g2v=loss_g2v.item())

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    if writer:
        writer.add_scalar('loss', metric_logger.loss.global_avg, step)
        writer.add_scalar('loss_g1v', metric_logger.loss_g1v.global_avg, step)
        writer.add_scalar('loss_g2v', metric_logger.loss_g2v.global_avg, step)
    return metric_logger.loss.global_avg


def main(args):
    global step

    utils.mkdir(args.output_path)
    train_writer, val_writer = None, None
    utils.init_distributed_mode(args)
    if args.tensorboard:
        if not args.distributed or (args.rank == 0) and (args.local_rank == 0):
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))
                                                                    
    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'illinois':
        data_min = -7.077488589857239e-06
        data_max = 9.758711712493096e-06
        label_min = 0.000123277
        label_max = 3.4196200370788574
    elif args.dataset == 'illinois_norm':
        data_min = -1.0
        data_max = 1.0
        label_min = 0.000123277
        label_max = 3.4196200370788574
    elif args.dataset == 'illinois_pure_field':
        data_min = -1.0
        data_max = 1.0
        label_min = -1.72
        label_max = 3.42#2.83
    elif args.dataset == 'illinois_heat_map':
        data_min = -1.0
        data_max = 1.0
        label_min = -0.01
        label_max = 1
    elif args.dataset == 'illinois_heat_map_3D':
        data_min = -1.0
        data_max = 1.0
        label_min = -0.008
        label_max = 0.806

    # Data loading code
    print('Loading data')

#    print('Loading field data')
#    field_data_0 = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_field_testing_set/field_ref_data_500_0.npy'))
#    field_data_1 = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_field_testing_set/field_ref_data_500_1.npy'))
#    field_data_2 = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_field_testing_set/field_ref_data_500_2.npy'))
#    field_data_3 = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_field_testing_set/field_ref_data_500_3.npy'))
#    field_data_4 = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_field_testing_set/field_ref_data_500_4.npy'))
#    field_data_5 = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_field_testing_set/field_ref_data_500_5.npy'))
#    field_data_6 = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_field_testing_set/field_ref_data_500_6.npy'))

#   field_data_all = torch.Tensor(np.load('./data/field_syn_data_bandpass_trace_norm_500_1_1000_31/field_data_picked.npy'))
#    field_data_all = torch.Tensor(np.load('/projects/piml_inversion/hwang/Illinois/src_mlreal/data/dataset_5_bandpass_no_swap/processed_syn_training_set/syn_data_500_0.npy'))

#    field_data_all = torch.cat((field_data_0,field_data_1,field_data_2,field_data_3,field_data_4,field_data_5,field_data_6),axis=0)
#    print('field data OK:',field_data_all.size)
    print('Loading training data')
    
    # Normalize data and label to [-1, 1]
    transform_data = Compose([
        T.LogTransform(k=args.k), # (legacy) log transformation
        T.MinMaxNormalize(T.log_transform(data_min, k=args.k), T.log_transform(data_max, k=args.k)) 
    ])
    transform_label = Compose([
        T.MinMaxNormalize(label_min, label_max)
    ])
    if args.train_anno[-3:] == 'txt':
        dataset_train = FWIDataset(
            args.train_anno,
            preload=True,
            sample_ratio=args.sample_ratio,
            file_size=args.file_size,
            transform_data=transform_data,
            transform_label=transform_label
        )
    else:
        dataset_train = torch.load(args.train_anno)

    print('Loading validation data')
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            preload=True,
            sample_ratio=args.sample_ratio,
            file_size=args.file_size,
            transform_data=transform_data,
            transform_label=transform_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
         
    if args.up_mode:    
        model = network.model_dict[args.model](upsample_mode=args.up_mode).to(device)
    else:
        model = network.model_dict[args.model]().to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v
    
    print('criterion OK!')

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size 
    #optimizer = torch.optim.SGD(
    #    model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    
    print('scheduler OK!')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    if args.resume: # load from checkpoint
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']

    print('Start training')
    start_time = time.time()
    best_loss = 100
    for epoch in range(args.start_epoch, args.epochs):
        print("current epoch:",epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train,
                        device, epoch, args.print_freq, train_writer)
        val_loss = evaluate(model, criterion, dataloader_valid, device, val_writer)
        lr_scheduler.step(val_loss)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'args': args}
        # Save checkpoint per epoch
        if val_loss < best_loss:
            utils.save_on_master(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
            print('saving checkpoint at epoch: ', epoch)
            chp = epoch
            best_loss = val_loss
        #utils.save_on_master(
        #    checkpoint,
        #    os.path.join(args.output_path, 'checkpoint.pth'))
        # Save checkpoint every epoch block
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('--anno-path', default='relevant_files', help='dataset files location')
    parser.add_argument('-t', '--train-anno', default='san_juan_time_lapes_make_models_train.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='san_juan_time_lapes_make_models_test.txt', help='name of val anno')
    parser.add_argument('-fs', '--file-size', default=500, type=int, help='samples per data file')
    parser.add_argument('-ds', '--dataset', default='SJTL', type=str, help='dataset option for normalization')
    parser.add_argument('-o', '--output-path', default='models', help='path where to save')
    parser.add_argument('-n', '--save-name', default='mlreal_illinois_pure_fd_nf', help='saved name for this run')
    parser.add_argument('-s', "--suffix", type=str, default=None)
    parser.add_argument('-m', '--model', help='select inverse model')
    parser.add_argument('--up_mode', default=None, help='upsample mode')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('-sr', '--sample_ratio', type=int, default=1, help='subsample ratio of data')
    parser.add_argument('-eb', '--epoch_block', type=int, default=160, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=4, help='number of saved block')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=1e-4 , type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    
    # distributed training parameters
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # tensorboard
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    
    args.epochs = args.epoch_block * args.num_block

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
