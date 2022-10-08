"""
可视化网络中间的特征图，网络的权重参数
@author: Wenwen Tong, Peking University
"""
import pdb
import torch
import torch.nn as nn
import numpy as np

class DataStructure():
    """"数据样本的结构类型"""
    def __init__(self, data, drag_coefficient, case_name, theory_cd, wake_dis=None, aoa=None, frequency=None):
        self.data = data
        self.wake_dis = wake_dis
        self.aoa = aoa
        self.drag_coefficient = drag_coefficient
        self.case_name = case_name
        self.theory_cd = theory_cd
        self.frequency = frequency

    def __repr__(self):
        return "the sample is "+self.case_name


class FeatureVisualization():
    """
    给定数据样本，可视化特征图，中间网络结构
    """
    def __init__(self, data_loader, net, visualize_path, args):
        self.data_loader = data_loader
        self.net = net
        self.visualize_path = visualize_path
        self.args = args
        self.data_index = 0

    def print_net_structure(self):
        print(self.net)

    def get_cur_data(self):
        for batch_index, data_list in enumerate(self.data_loader):
            frequency = None
            aoa = None
            wake_dis = None

            if self.args.dataset == 'stationary_plate':
                data, wake_dis, aoa, drag_coefficient, case_name, theory_cd = data_list
            elif self.args.dataset == 'flapping_plate':
                data, wake_dis, aoa, drag_coefficient, case_name, theory_cd, frequency = data_list

            cur_size = len(data)
            print('batch size:', cur_size)
            drag_coefficient = drag_coefficient.float()
            theory_cd = theory_cd.float()
            aoa = aoa.float()
            wake_dis = wake_dis.float()
            if self.args.dataset == 'flapping_plate':
                frequency = frequency.float()

            if self.args.gpu:  
                data = data.cuda()
                aoa = aoa.cuda()
                drag_coefficient = drag_coefficient.cuda()
                wake_dis = wake_dis.cuda()
                if self.args.dataset == 'flapping_plate':
                    frequency = frequency.cuda()

            data_structure = DataStructure(data, drag_coefficient, case_name, theory_cd,
                                           wake_dis=wake_dis, aoa=aoa, frequency=frequency)
            return data_structure


    @torch.no_grad()
    def get_feature(self):
        self.net.eval()
        data_structure = self.get_cur_data()
        input_data = data_structure.data
        aoa = data_structure.aoa
        wake_dis = data_structure.wake_dis
        drag_coefficient = data_structure.drag_coefficient
        case_name = data_structure.case_name
        theory_cd = data_structure.theory_cd
        fre = data_structure.frequency

        res_dict = {}
        x = input_data
        prediction = self.net(input_data, aoa=aoa, wake_dis=wake_dis)
        for layer_name, layer in self.net.named_children():
            if 'conv' in layer_name:
                print(layer_name)
                x = layer(x)
                # 获取最后一层的features
                if layer_name == 'conv5_x':
                    last_features = x

            elif layer_name == 'avg_pool':
                print(layer_name)
                x = layer(x)
                x = x.view(x.size(0), -1)

            elif layer_name == 'fc':
                print(layer_name)
                x = layer(x)

            elif layer_name == 'fc_aoa' and self.net.add_aoa and aoa is not None:
                print(layer_name)
                aoa = aoa.unsqueeze(1)  
                output_aoa = layer(aoa)
                x = x + output_aoa

            elif layer_name == 'fc_wake_dis' and self.net.add_wake_dis and wake_dis is not None:
                print(layer_name)
                wake_dis = wake_dis.unsqueeze(1)
                output_wake_dis = layer(wake_dis)
                x = x + output_wake_dis

            elif layer_name == 'fc_fre' and self.net.add_fre and fre is not None:
                print(layer_name)
                fre = fre.unsqueeze(1)
                output_fre = layer(fre)
                x = x + output_fre

            elif layer_name == 'fc_force':
                print(layer_name)
                x = layer(x)

            y = x.numpy()
            res_dict[layer_name] = y


        gt_drag_coefficient = drag_coefficient.squeeze().numpy()
        gt_wake_dis = wake_dis.squeeze().numpy()
        gt_case_name = list(case_name)
        gt_aoa = aoa.squeeze().numpy()

        res_dict['input_data'] = input_data.numpy()
        res_dict['gt_drag_coefficient'] = gt_drag_coefficient
        res_dict['gt_wake_dis'] = gt_wake_dis
        res_dict['gt_case_name'] = gt_case_name
        res_dict['gt_aoa'] = gt_aoa
        res_dict['pre_drag_coefficient'] = x.squeeze().numpy()

        return res_dict


    def cam(self):
        self.net.eval()
        print('cam')
        data_structure = self.get_cur_data()
        input_data = data_structure.data
        aoa = data_structure.aoa
        wake_dis = data_structure.wake_dis
        drag_coefficient = data_structure.drag_coefficient
        fre = data_structure.frequency

        x = input_data
        for layer_name, layer in self.net.named_children():
            if 'conv' in layer_name:
                x = layer(x)
                # 获取最后一层的features
                if layer_name == 'conv5_x':
                    last_features = x

            elif layer_name == 'avg_pool':
                x = layer(x)
                x = x.view(x.size(0), -1)

            elif layer_name == 'fc':
                x = layer(x)

            elif layer_name == 'fc_aoa' and self.net.add_aoa and aoa is not None:
                aoa = aoa.unsqueeze(1)  
                output_aoa = layer(aoa)
                x = x + output_aoa

            elif layer_name == 'fc_wake_dis' and self.net.add_wake_dis and wake_dis is not None:
                wake_dis = wake_dis.unsqueeze(1)
                output_wake_dis = layer(wake_dis)
                x = x + output_wake_dis

            elif layer_name == 'fc_fre' and self.net.add_fre and fre is not None:
                fre = fre.unsqueeze(1)
                output_fre = layer(fre)
                x = x + output_fre

            elif layer_name == 'fc_force':
                x = layer(x)

        """
        CAM 类激活映射: 最后一个卷积层的CAM
        """
        # 为了能读取到中间梯度定义的辅助函数
        def extract(g):
            global features_grad
            features_grad = g

        loss_function = nn.MSELoss(reduction='mean')
        if self.args.gpu:
            loss_function = loss_function.cuda()
        outputs = x.squeeze()
        loss = loss_function(outputs, drag_coefficient)
        last_features.register_hook(extract)
        loss.backward()
        grads = features_grad  # 获取梯度
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))  # shape: [32, 512, 1, 1]
        batch_size, channel, _ , _ = pooled_grads.shape
        for i in range(batch_size):
            for j in range(channel):
                last_features[i,j,...] = last_features[i,j,...]*pooled_grads[i,j,...]
        heatmap = last_features.detach().numpy()   # 涉及到梯度传播的tensor变量不能够直接numpy()转numpy, 需要首先进行detach操作


        return heatmap
