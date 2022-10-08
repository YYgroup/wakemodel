""" train network using pytorch

@author: Wenwen Tong, Peking University
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pdb

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_logger
from dataset import CIFAR100Train, CIFAR100Test, StationaryPlateWakeData, StationaryPlateDataset, FlappingPlateWakeData, FlappingPlateDataset
from postprocess.visualize_feature_map import  FeatureVisualization
from collections import defaultdict

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, data_list in enumerate(train_loader):

        if args.dataset == 'stationary_plate':
            data, wake_dis, aoa, drag_coefficient, case_name, theory_cd = data_list
        elif args.dataset == 'flapping_plate':
            data, wake_dis, aoa, drag_coefficient, case_name, theory_cd, frequency = data_list
        
        drag_coefficient = drag_coefficient.float()
        theory_cd = theory_cd.float()
        aoa = aoa.float()
        wake_dis = wake_dis.float()
        if args.dataset == 'flapping_plate':
            frequency = frequency.float()

        if args.gpu:  # 数据需要传到gpu中
            data = data.cuda()
            aoa = aoa.cuda()
            drag_coefficient = drag_coefficient.cuda()
            wake_dis = wake_dis.cuda()
            if args.dataset == 'flapping_plate':
                frequency = frequency.cuda()

        optimizer.zero_grad()
        if args.aoa:
            outputs = net(data, aoa=aoa)
        elif args.fre:
            outputs = net(data, fre = frequency)
        elif args.aoa_fre:
            outputs = net(data, aoa=aoa, fre = frequency)
        else:
            outputs = net(data)

        
        outputs = outputs.squeeze()  #默认输出只有受力
        loss = loss_function(outputs, drag_coefficient)

        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]

        if args.debug:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.8f}\tLR: {:0.8f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.train_batch + len(data),
                total_samples=len(train_loader.dataset)
            ))
        else:
            logger.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.8f}\tLR: {:0.8f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.train_batch + len(data),
                total_samples=len(train_loader.dataset)
            ))

        #update training loss for each iteration
        if not args.only_validation and not args.debug:
            writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    if not args.only_validation and not args.debug:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()
    if args.debug:
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    else:
        logger.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True, save_pre=False):
    """tb: save data to tensorboard
       save_pre: save model prediction result
    """
    start = time.time()
    net.eval()

    total_loss = 0.0 # cost function error
    average_loss = 0.0
    correct = 0.0
    total_force_loss =0.0
    total_wake_loss = 0.0
    total_aoa_loss = 0.0

    pre_drag_coefficient = []
    pre_wake_dis = []
    pre_plate_aoa = []
    gt_drag_coefficient = []
    gt_wake_dis = []
    gt_aoa = []
    gt_case_name = []
    print('total_samples in validation:', len(test_loader.dataset))
    for batch_index, data_list in enumerate(test_loader):

        if args.dataset == 'stationary_plate':
            data, wake_dis, aoa, drag_coefficient, case_name, theory_cd = data_list
        elif args.dataset == 'flapping_plate':
            data, wake_dis, aoa, drag_coefficient, case_name, theory_cd, frequency = data_list

        cur_size = len(data)
        drag_coefficient = drag_coefficient.float()
        theory_cd = theory_cd.float()
        aoa = aoa.float()
        wake_dis = wake_dis.float()
        if args.dataset == 'flapping_plate':
            frequency = frequency.float()

        if args.gpu:  # 数据需要传到gpu中
            data = data.cuda()
            aoa = aoa.cuda()
            drag_coefficient = drag_coefficient.cuda()
            wake_dis = wake_dis.cuda()
            if args.dataset == 'flapping_plate':
                frequency = frequency.cuda()

        if args.aoa:
            outputs = net(data, aoa=aoa)
        elif args.fre:
            outputs = net(data, fre = frequency)
        elif args.aoa_fre:
            outputs = net(data, aoa=aoa, fre = frequency)
        else:
            outputs = net(data) # outputs.size() : torch.Size([batch_size, 1])
        
        outputs = outputs.squeeze()
        loss = loss_function(outputs, drag_coefficient)
        pre_drag_coefficient += outputs.tolist()

        total_loss = total_loss+loss.item()*cur_size

        gt_drag_coefficient += drag_coefficient.tolist()
        gt_wake_dis += wake_dis.tolist()
        gt_case_name += list(case_name)
        gt_aoa += aoa.tolist()

    # average loss
    average_loss = total_loss/len(test_loader.dataset)

    # 转换为numpy保存到输出结果中
    pre_drag_coefficient = np.array(pre_drag_coefficient)
    gt_drag_coefficient = np.array(gt_drag_coefficient)
    gt_wake_dis = np.array(gt_wake_dis)
    gt_aoa = np.array(gt_aoa)
    gt_case_name = np.array(gt_case_name)


    if save_pre:  # 保存数据
        print('save validation set result')
        npz_name = 'test_loader_data_'+str(epoch)+'.npz'
        
        if args.save_pre_path:
            save_path = os.path.join(args.save_pre_path, npz_name)
        else:
            save_path = os.path.join(log_dir, npz_name)
        np.savez(save_path, gt_case_name = gt_case_name,
                 pre_drag = pre_drag_coefficient, 
                 gt_drag = gt_drag_coefficient, gt_wake_dis = gt_wake_dis,
                 gt_aoa = gt_aoa)

        ### 直接添加预测结果
        def cal_err(force_pre, force_dns, positions):
            positions = np.array(positions)
            force_pre = np.array(force_pre)
            force_dns = np.array(force_dns)
            error = (force_pre - force_dns) / force_dns
            error = np.absolute(error)[(positions < 15) & (positions > 5)]

            mean_error = np.mean(error)
            return mean_error

        test_data = np.load(save_path, allow_pickle=True)
        pre_drag_coefficient = test_data['pre_drag']
        gt_drag_coefficient = test_data['gt_drag']
        gt_wake_dis = test_data['gt_wake_dis'] * 10
        gt_case_name = test_data['gt_case_name']

        print('sample number:', len(gt_case_name))
        case_names = np.unique(gt_case_name)

        res = {}  # 根据每个case设置一个dict文件保存结果
        for i in range(len(case_names)):
            cur_case = case_names[i]
            res[cur_case] = {}
            res[cur_case]['pre_drag_coefficient'] = pre_drag_coefficient[gt_case_name == cur_case]
            res[cur_case]['gt_drag_coefficient'] = gt_drag_coefficient[gt_case_name == cur_case]
            res[cur_case]['gt_wake_dis'] = gt_wake_dis[gt_case_name == cur_case]

        mean_errors = []
        for case in res:
            pre_drag_coefficient = res[case]['pre_drag_coefficient']
            gt_drag_coefficient = res[case]['gt_drag_coefficient']
            gt_wake_dis = res[case]['gt_wake_dis']
            order = np.argsort(gt_wake_dis)
            gt_wake_dis = gt_wake_dis[order]
            pre_drag_coefficient = pre_drag_coefficient[order]

            mean_error = cal_err(pre_drag_coefficient, gt_drag_coefficient, gt_wake_dis)
            print('case:', case, 'mean force error:', mean_error)
            mean_errors.append(mean_error)
        print('total mean errors:', np.mean(mean_errors))
        ###########################################################################

    finish = time.time()

    if args.gpu:
        print('GPU INFO.....')
        # print(torch.cuda.memory_summary(), end='')  ## AttributeError: module 'torch.cuda' has no attribute 'memory_summary'
    print('Evaluating Network.....')
    
    print('Test set: Epoch: {}, Average loss: {:.8f}, Time consumed:{:.2f}s'.format(epoch, average_loss, finish - start))
    logger.info('Test set: Epoch: {}, Average loss: {:.8f}, Time consumed:{:.2f}s'.format(epoch, average_loss, finish - start))
    print()

    #add informations to tensorboard
    if not args.only_validation and not args.debug:
        if tb:
            writer.add_scalar('Test/Average_loss', average_loss, epoch)
            writer.add_scalar('Test/Average_force_loss', total_force_loss/len(test_loader.dataset), epoch)
            writer.add_scalar('Test/Average_aoa_loss', total_aoa_loss / len(test_loader.dataset), epoch)
    return average_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-debug', action='store_true', default=False, help='debug of not')
    parser.add_argument('-train_batch', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-dataset', type=str, default='stationary_plate', help='dataset type')
    parser.add_argument('-data_path', type=str, default='./data/cifar-100-python', help='dataset path')
    parser.add_argument('-only_validation', action='store_true', default=False, help='only_validation')
    parser.add_argument('-weight_path', type=str, default='resnet18-100-regular.pth', help='weight_path')
    parser.add_argument('-save_pre_path', type=str, default='', help='save data path')

    parser.add_argument('-save_pre', action='store_true', default=False, help='whether save pre during training')
    parser.add_argument('-train_val_type', type=int, default=1, help='the split of train and validation set')
    parser.add_argument('-exp_name', type=str, default='', help='experiment name')

    parser.add_argument('-img_size', type=int, default=64, help='img_size for the network input')
    parser.add_argument('-not_revise_input_data', action='store_true', default=False, help='whether revise_npy_file')
    
    #insert additional information
    parser.add_argument('-aoa', action='store_true',default=False, help='insert aoa in the fc layers')
    parser.add_argument('-fre', action='store_true', default=False, help='insert fre in the fc layers')
    parser.add_argument('-aoa_fre', action='store_true', default=False, help='insert aoa  and fre in the fc layers')
    parser.add_argument('-visualize_feature', action='store_true', default=False, help='whether visualize feature of network')

    # add noise to velocity data
    parser.add_argument('-add_noise', action='store_true', default=False,  help='whether add_noise for the velocity data')
    parser.add_argument('-sigma', type=float, default=0.05, help='loss weight for force')

    # 使用平板数据集来训练，使用椭圆合页数据来验证
    # parser.add_argument('-train_dataset_type', type=str, default='', help='dataset type, st or fl or st_fl')
    # parser.add_argument('-st_train_type', type=int, default=10, help='the train data of stationary plate')
    # parser.add_argument('-st_data_path', type=str, default='./data', help='dataset path')
    # parser.add_argument('-fl_train_type', type=int, default=10, help='the train data of flapping plate')
    # parser.add_argument('-fl_data_path', type=str, default='./data', help='dataset path')

    # parser.add_argument('-val_dataset', type=str, default='ellipse', help='dataset type')
    # parser.add_argument('-val_type', type=int, default=1, help='the split of train and validation set')
    # parser.add_argument('-val_data_path', type=str, default='./data', help='dataset path')

    args = parser.parse_args()
    print('=========args===============')
    print(args)
    net = get_network(args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        args.gpu = True
        net.to(device)  
    else:
        device = torch.device("cpu")

    print('model in device', device)
    print('args.gpu:', args.gpu)

    loss_function = nn.MSELoss(reduction='mean')  
    loss_force_function = nn.MSELoss(reduction='mean')
    loss_aoa_function = nn.MSELoss(reduction='mean')
    loss_wake_dis_function = nn.MSELoss(reduction='mean')
    loss_theory_function = nn.MSELoss(reduction='mean')
    if args.gpu:
        loss_function = loss_force_function.cuda()
        loss_force_function = loss_force_function.cuda()
        loss_wake_dis_function = loss_wake_dis_function.cuda()
        loss_aoa_function = loss_aoa_function.cuda()
        loss_theory_function = loss_theory_function.cuda()


    if args.dataset == 'stationary_plate' and not args.train_dataset_type :
        plate_data = StationaryPlateWakeData(args.data_path, train_val_type=args.train_val_type,
                                             revise_npy_file=not args.not_revise_input_data,
                                             add_noise = args.add_noise, sigma = args.sigma)  # generate dataset
        train_dict, test_dict = plate_data.generate_data_dict()
        train_dataset = StationaryPlateDataset(train_dict, img_size=args.img_size)
        train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=args.train_batch)
        test_dataset = StationaryPlateDataset(test_dict, img_size=args.img_size)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=args.train_batch)
        print('finish generating stationary plate train and test loader')

    # elif args.dataset == 'flapping_plate':
    #     plate_data = FlappingPlateWakeData(args.data_path, train_val_type=args.train_val_type,
    #                                        add_noise = args.add_noise, sigma = args.sigma)  # generate dataset
    #     train_dict, test_dict = plate_data.generate_data_dict()
    #     train_dataset = FlappingPlateDataset(train_dict, img_size=args.img_size)
    #     train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=args.train_batch)
    #     test_dataset = FlappingPlateDataset(test_dict, img_size=args.img_size)
    #     test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=args.train_batch)
    #     print('finish generating flapping plate train and test loader')

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    exp_path = 'experiment'  
    os.makedirs(exp_path, exist_ok=True)
    if args.dataset == 'stationary_plate':
        exp_plate_path = os.path.join(exp_path, 'stationary_plate')
    elif args.dataset == 'flapping_plate':
        exp_plate_path = os.path.join(exp_path, 'flapping_plate')
    os.makedirs(exp_plate_path, exist_ok=True)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW+args.exp_name)
        cur_exp_path = os.path.join(exp_plate_path, args.exp_name+'_'+args.net)
        log_dir = cur_exp_path
        if not os.path.exists(cur_exp_path):
            os.makedirs(cur_exp_path)
        checkpoint_path = os.path.join(cur_exp_path, '{net}-{epoch}-{type}.pth')

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):  # runs文件夹
        os.mkdir(settings.LOG_DIR)

    #visualize feature map
    if args.visualize_feature:
        print('visualize feature map in the network')
        if args.gpu:
            net.load_state_dict(torch.load(args.weight_path))
        else:
            net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
        visualize_path = os.path.join(cur_exp_path, 'visualize_map')
        os.makedirs(visualize_path, exist_ok=True)
        extractor  = FeatureVisualization(test_loader, net, visualize_path, args)
        res_dict = extractor.get_feature()
        heatmap = extractor.cam()
        res_dict['heatmap'] = heatmap
        cur_file_name = 'data_dict.npy'
        cur_file_path = os.path.join(visualize_path, cur_file_name)
        np.save(cur_file_path, res_dict)
        sys.exit()  

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    if not args.only_validation and not args.debug:   
        writer = SummaryWriter(log_dir=log_dir)  
        logger_file_name = 'exp.log'  
        logger_file_path = os.path.join(log_dir, logger_file_name)
        logger = get_logger(logger_file_path)

        input_tensor = torch.Tensor(1, 3, 32, 32)
        if args.gpu:
            input_tensor = input_tensor.cuda()
        # writer.add_graph(net, input_tensor)

    minimum_loss = 1.0

    # if args.only_validation:
    #     net.load_state_dict(torch.load(args.weight_path))
    #     print('==start validation==')
    #     # loss = eval_training(tb=False, save_pre=True)
    #     loss = eval_training(tb=False, save_pre=args.save_pre)
    #     print('==end validation==')
    #     sys.exit()  


    if args.resume:
        best_weights = best_acc_weights(cur_exp_path)
        if best_weights:
            weights_path = os.path.join(cur_exp_path, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(cur_exp_path)
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(cur_exp_path, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(cur_exp_path)

    # train and eval
    if not args.debug:
        logger.info('start training!')
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        if args.debug:
            sys.exit()

        cur_loss = eval_training(epoch, save_pre=args.save_pre)

        if cur_loss < minimum_loss:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

            minimum_loss = cur_loss
            print('best epoch:', epoch, 'minimum_loss:', minimum_loss)
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)


    logger.info('end training!')
    writer.close()
