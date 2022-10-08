""" train and test dataset

@author: Wenwen Tong, Peking University
"""
import os
import sys
import pickle
import torch

from skimage import io
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import  get_stationary_plate_theory_force, get_flapping_plate_theory_force, generate_noise_data
import pdb
import glob

class StationaryPlateWakeData:
    """StationaryPlateWake dataset
    args:
        train_val_type: split the data into the training set and validation set
        revise_npy_file: normalize the input velocity data
        add_noise: noising the data
        sigma: noise to signal ratio
    """

    def __init__(self, path, train_val_type=1,transform=None, revise_npy_file=True,
                 add_noise = False, sigma = 0.05):
        self.root_path = path
        self.drag_coefficient_dict ={
            're300_aoa5_ar2': 0.234,
            're300_aoa10_ar2': 0.267,
            're300_aoa15_ar2': 0.314,
            're300_aoa20_ar2': 0.360,
            're300_aoa25_ar2': 0.427,
            're300_aoa30_ar2': 0.498,
            're300_aoa35_ar2': 0.584,
            're300_aoa40_ar2': 0.661
        }
        self.train_dict = {}
        self.test_dict = {}
        self.normalize_wake_dis = True
        self.wake_dis_max = 10.0
        print('revise_npy_file:', revise_npy_file)
        print('train_val_type=', train_val_type)
        print('add_noise:', add_noise, ' noise intensity:', sigma)

        if train_val_type == 1:
            self.train_cases = ['re300_aoa5_ar2', 're300_aoa10_ar2', 're300_aoa35_ar2', 're300_aoa40_ar2']
            self.test_cases = ['re300_aoa15_ar2', 're300_aoa20_ar2', 're300_aoa25_ar2', 're300_aoa30_ar2']
        elif train_val_type == 2:
            self.train_cases = ['re300_aoa15_ar2', 're300_aoa20_ar2', 're300_aoa25_ar2', 're300_aoa30_ar2']
            self.test_cases = ['re300_aoa5_ar2', 're300_aoa10_ar2', 're300_aoa35_ar2', 're300_aoa40_ar2']
        elif train_val_type == 3:
            self.train_cases = ['re300_aoa5_ar2', 're300_aoa10_ar2', 're300_aoa15_ar2',
                                're300_aoa30_ar2', 're300_aoa35_ar2', 're300_aoa40_ar2']
            self.test_cases = ['re300_aoa20_ar2', 're300_aoa25_ar2']
        else:
            raise NotImplementedError

        self.train_data = []
        self.train_reynolds_number = []
        self.train_aspect_ratio = []
        self.train_aoa = []
        self.train_wake_dis = []
        self.train_drag_coefficient = []
        self.train_lift_coefficient = []
        self.train_case_name = []
        self.train_theory_cd = []
        self.train_shedding = []

        self.test_data = []
        self.test_reynolds_number = []
        self.test_aspect_ratio = []
        self.test_aoa = []
        self.test_wake_dis = []
        self.test_drag_coefficient = []
        self.test_lift_coefficient = []
        self.test_case_name = []
        self.test_theory_cd = []
        self.test_shedding = []

        # read train data
        print('===read train data')
        for case in self.train_cases:
            print('read data:', case)
            case_dir = os.path.join(path, case)
            case_list = case.split('_')
            reynolds_number = int(case_list[0][2:])
            aoa = float(case_list[1][3:])*np.pi/180
            if aoa < 23:
                shedding = False
            else:
                shedding = True
            aspect_ratio = float(case_list[2][2:])
            for file_name in os.listdir(case_dir):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(case_dir, file_name)
                    wake_dis = float(file_name.split('_')[3])
                    # choose the sample in the wake
                    if abs(wake_dis -1) > 0.05 and abs(wake_dis -4) > 0.05 and abs(wake_dis -8) > 0.05 and abs(wake_dis -12.8) > 0.05:
                        if self.normalize_wake_dis:
                            wake_dis = wake_dis/self.wake_dis_max
                        cur_data = np.load(file_path).astype('float32')  # (H, W, C)
                        if add_noise:
                            cur_data = generate_noise_data(cur_data, add_noise=True, sigma = sigma)
                        
                        theory_force = get_stationary_plate_theory_force(cur_data)
                        theory_cd = theory_force

                        if revise_npy_file:
                            cur_max = np.max(cur_data)
                            cur_min = np.min(cur_data)
                            cur_data = (cur_data - cur_min) / (cur_max - cur_min)
                            
                        height, width, _ = cur_data.shape
                        cur_data = cur_data.transpose(2, 0, 1)  # (C, H, W)
                        cur_data = torch.from_numpy(cur_data).float()

                        self.train_data.append(cur_data)
                        self.train_reynolds_number.append(reynolds_number)
                        self.train_aoa.append(aoa)
                        self.train_aspect_ratio.append(aspect_ratio)

                        self.train_wake_dis.append(wake_dis)
                        self.train_drag_coefficient.append(self.drag_coefficient_dict[case])
                        self.train_case_name.append(case)
                        self.train_theory_cd.append(theory_cd)
                        if shedding:
                            self.train_shedding.append(1)
                        else:
                            self.train_shedding.append(0)


        # read test data
        print('===read test data')
        for case in self.test_cases:
            print('read data:', case)
            case_dir = os.path.join(path, case)
            case_list = case.split('_')
            reynolds_number = int(case_list[0][2:])
            aoa = float(case_list[1][3:])*np.pi/180
            if aoa < 23:
                shedding = False
            else:
                shedding = True
            aspect_ratio = float(case_list[2][2:])
            for file_name in os.listdir(case_dir):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(case_dir, file_name)
                    wake_dis = float(file_name.split('_')[3])
                    # choose the sample in the wake
                    if abs(wake_dis -1) > 0.05 and abs(wake_dis -4) > 0.05 and abs(wake_dis -8) > 0.05 and abs(wake_dis -12.8) > 0.05:
                        if self.normalize_wake_dis:
                            wake_dis = wake_dis/self.wake_dis_max
                        cur_data = np.load(file_path).astype('float32')
                        if add_noise:
                            cur_data = generate_noise_data(cur_data, add_noise=True, sigma = sigma)
                        theory_force = get_stationary_plate_theory_force(cur_data)
                        theory_cd = theory_force
                        if revise_npy_file:
                            cur_max = np.max(cur_data)
                            cur_min = np.min(cur_data)
                            cur_data = (cur_data - cur_min) / (cur_max - cur_min)

                        height, width,_ = cur_data.shape
                        
                        cur_data = cur_data.transpose(2, 0, 1)
                        cur_data = torch.from_numpy(cur_data).float()

                        self.test_data.append(cur_data)
                        self.test_reynolds_number.append(reynolds_number)
                        self.test_aoa.append(aoa)
                        self.test_aspect_ratio.append(aspect_ratio)

                        self.test_wake_dis.append(wake_dis)
                        self.test_drag_coefficient.append(self.drag_coefficient_dict[case])
                        self.test_case_name.append(case)
                        self.test_theory_cd.append(theory_cd)
                        if shedding:
                            self.test_shedding.append(1)
                        else:
                            self.test_shedding.append(0)

    def generate_data_dict(self):
        self.train_dict['data'] = self.train_data
        self.train_dict['aoa'] = self.train_aoa
        self.train_dict['drag_coefficient'] = self.train_drag_coefficient
        self.train_dict['wake_dis'] = self.train_wake_dis
        self.train_dict['case_name'] = self.train_case_name
        self.train_dict['theory_cd'] = self.train_theory_cd
        self.train_dict['shedding'] = self.train_shedding

        self.test_dict['data'] = self.test_data
        self.test_dict['aoa'] = self.test_aoa
        self.test_dict['drag_coefficient'] = self.test_drag_coefficient
        self.test_dict['wake_dis'] = self.test_wake_dis
        self.test_dict['case_name'] = self.test_case_name
        self.test_dict['theory_cd'] = self.test_theory_cd
        self.test_dict['shedding'] = self.test_shedding

        return self.train_dict, self.test_dict

class StationaryPlateDataset(Dataset):
    """plate_train_dataset
    """
    def __init__(self, data_dict, img_size=64, transform=None):
        self.data_dict = data_dict
        self.data = self.data_dict['data']
        self.aoa = self.data_dict['aoa']
        self.wake_dis = self.data_dict['wake_dis']
        self.drag_coefficient = self.data_dict['drag_coefficient']
        self.case_name = self.data_dict['case_name']
        self.theory_cd = self.data_dict['theory_cd']

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        wake_dis = self.wake_dis[index]
        aoa = self.aoa[index]
        drag_coefficient = self.drag_coefficient[index]
        case_name = self.case_name[index]
        theory_cd = self.theory_cd[index]

        if self.transform:
            data = self.transform(data)
        return data, wake_dis, aoa, drag_coefficient, case_name, theory_cd

"""
class FlappingPlateWakeData:
    """Flapping PlateWake dataset
    """

    def __init__(self, path, train_val_type=1,transform=None, revise_npy_file=True,
                 add_noise = False, sigma = 0.05):
        self.root_path = path
        self.drag_coefficient_dict ={
            'case1_ar1_f_0.6_alpha_30': -0.765,
            'case2_ar1_f_0.6_alpha_15': -0.770,
            'case3_ar1_f_0.6_alpha_45': -0.313,
            'case4_ar1_f_0.7_alpha_15': -1.230,
            'case5_ar1_f_0.7_alpha_30': -1.303,
            'case6_ar1_f_0.7_alpha_45': -0.744,
            'case7_ar1_f_0.8_alpha_15': -1.767,
            'case8_ar1_f_0.8_alpha_30': -1.960,
            'case9_ar1_f_0.8_alpha_45': -1.295,
            'case10_ar1_f_0.65_alpha_22.5': -1.091,
            'case11_ar1_f_0.65_alpha_37.5': -0.812,
            'case12_ar1_f_0.75_alpha_22.5': -1.676,
            'case13_ar1_f_0.75_alpha_37.5': -1.370,
        }
        self.train_dict = {}
        self.test_dict = {}
        self.normalize_wake_dis = True
        self.wake_dis_max = 10.0  # 扑动平板尾迹更长

        self.cases = list(self.drag_coefficient_dict.keys())
        print('train_val_type=', train_val_type)
        print('add_noise:', add_noise, ' noise intensity:', sigma)

        if train_val_type == 1:
            self.train_cases = ['case2_ar1_f_0.6_alpha_15', 'case3_ar1_f_0.6_alpha_45','case5_ar1_f_0.7_alpha_30',
                                'case7_ar1_f_0.8_alpha_15', 'case9_ar1_f_0.8_alpha_45']
            self.test_cases = ['case1_ar1_f_0.6_alpha_30', 'case4_ar1_f_0.7_alpha_15',
                               'case6_ar1_f_0.7_alpha_45', 'case8_ar1_f_0.8_alpha_30']
        elif train_val_type == 2:
            self.train_cases = ['case1_ar1_f_0.6_alpha_30', 'case2_ar1_f_0.6_alpha_15', 'case3_ar1_f_0.6_alpha_45',
                                'case4_ar1_f_0.7_alpha_15', 'case6_ar1_f_0.7_alpha_45', 'case7_ar1_f_0.8_alpha_15',
                                'case8_ar1_f_0.8_alpha_30', 'case9_ar1_f_0.8_alpha_45'
                                ]
            self.test_cases = ['case5_ar1_f_0.7_alpha_30']
        elif train_val_type == 3:
            self.train_cases = ['case1_ar1_f_0.6_alpha_30', 'case2_ar1_f_0.6_alpha_15', 'case3_ar1_f_0.6_alpha_45',
                                'case4_ar1_f_0.7_alpha_15', 'case5_ar1_f_0.7_alpha_30', 'case6_ar1_f_0.7_alpha_45',
                                'case7_ar1_f_0.8_alpha_15', 'case8_ar1_f_0.8_alpha_30', 'case9_ar1_f_0.8_alpha_45'
                                ]
            self.test_cases = ['case10_ar1_f_0.65_alpha_22.5', 'case11_ar1_f_0.65_alpha_37.5',
                               'case12_ar1_f_0.75_alpha_22.5', 'case13_ar1_f_0.75_alpha_37.5']
        else:
            raise NotImplementedError

        self.train_data = []
        self.train_reynolds_number = []
        self.train_aspect_ratio = []
        self.train_aoa = []
        self.train_frequency = []
        self.train_wake_dis = []
        self.train_drag_coefficient = []
        self.train_lift_coefficient = []
        self.train_case_name = []
        self.train_theory_cd = []

        self.test_data = []
        self.test_reynolds_number = []
        self.test_aspect_ratio = []
        self.test_aoa = []
        self.test_frequency = []
        self.test_wake_dis = []
        self.test_drag_coefficient = []
        self.test_lift_coefficient = []
        self.test_case_name = []
        self.test_theory_cd = []

        # read train data
        print('===read train data')
        for case in self.train_cases:
            print('read data:', case)
            case_dir = os.path.join(path, case)
            case_list = case.split('_')
            frequency = float(case_list[3])
            aoa = float(case_list[-1])*np.pi/180
            aspect_ratio = 1.0
            reynolds_number= 200
            for file_name in os.listdir(case_dir):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(case_dir, file_name)
                    wake_dis = float(file_name.split('_')[3])
                    # choose the sample in the wake
                    if abs(wake_dis - 2) > 0.05 and abs(wake_dis - 4) > 0.05 and abs(wake_dis - 8) > 0.05 and abs( wake_dis - 12) > 0.05:
                        if self.normalize_wake_dis:
                            wake_dis = wake_dis/self.wake_dis_max
                        cur_data = np.load(file_path).astype('float32')  # (H, W, C)
                        if add_noise:
                            cur_data = generate_noise_data(cur_data, add_noise=True, sigma = sigma)

                        theory_force = get_flapping_plate_theory_force(cur_data)
                        theory_cd = theory_force*2

                        if revise_npy_file:
                            cur_max = np.max(cur_data)
                            cur_min = np.min(cur_data)
                            cur_data = (cur_data - cur_min) / (cur_max - cur_min)

                        cur_data = cur_data.transpose(2, 0, 1)
                        cur_data = torch.from_numpy(cur_data).float()

                        self.train_data.append(cur_data)
                        self.train_reynolds_number.append(reynolds_number)
                        self.train_aoa.append(aoa)
                        self.train_aspect_ratio.append(aspect_ratio)
                        self.train_frequency.append(frequency)

                        self.train_wake_dis.append(wake_dis)
                        self.train_drag_coefficient.append(self.drag_coefficient_dict[case])
                        self.train_case_name.append(case)
                        self.train_theory_cd.append(theory_cd)

        # read test data
        print('===read test data')
        for case in self.test_cases:
            print('read data:', case)
            case_dir = os.path.join(path, case)
            case_list = case.split('_')
            frequency = float(case_list[3])
            aoa = float(case_list[-1]) * np.pi / 180
            aspect_ratio = 1.0
            reynolds_number = 200
            for file_name in os.listdir(case_dir):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(case_dir, file_name)
                    wake_dis = float(file_name.split('_')[3])
                    # choose the sample in the wake
                    if abs(wake_dis - 2) > 0.05 and abs(wake_dis - 4) > 0.05 and abs(wake_dis - 8) > 0.05 and abs( wake_dis - 12) > 0.05:
                        if self.normalize_wake_dis:
                            wake_dis = wake_dis/self.wake_dis_max
                        cur_data = np.load(file_path).astype('float32')
                        if add_noise:
                            cur_data = generate_noise_data(cur_data, add_noise=True, sigma = sigma)

                        theory_force = get_flapping_plate_theory_force(cur_data)
                        theory_cd = theory_force * 2
                        if revise_npy_file:
                            cur_max = np.max(cur_data)
                            cur_min = np.min(cur_data)
                            cur_data = (cur_data - cur_min) / (cur_max - cur_min)

                        cur_data = cur_data.transpose(2, 0, 1)
                        cur_data = torch.from_numpy(cur_data).float()

                        self.test_data.append(cur_data)
                        self.test_reynolds_number.append(reynolds_number)
                        self.test_aoa.append(aoa)
                        self.test_aspect_ratio.append(aspect_ratio)
                        self.test_frequency.append(frequency)

                        self.test_wake_dis.append(wake_dis)
                        self.test_drag_coefficient.append(self.drag_coefficient_dict[case])
                        self.test_case_name.append(case)
                        self.test_theory_cd.append(theory_cd)

    def generate_data_dict(self):
        self.train_dict['data'] = self.train_data
        self.train_dict['aoa'] = self.train_aoa
        self.train_dict['drag_coefficient'] = self.train_drag_coefficient
        self.train_dict['wake_dis'] = self.train_wake_dis
        self.train_dict['case_name'] = self.train_case_name
        self.train_dict['frequency'] = self.train_frequency
        self.train_dict['theory_cd'] = self.train_theory_cd

        self.test_dict['data'] = self.test_data
        self.test_dict['aoa'] = self.test_aoa
        self.test_dict['drag_coefficient'] = self.test_drag_coefficient
        self.test_dict['wake_dis'] = self.test_wake_dis
        self.test_dict['case_name'] = self.test_case_name
        self.test_dict['frequency'] = self.test_frequency
        self.test_dict['theory_cd'] = self.test_theory_cd

        return self.train_dict, self.test_dict


class FlappingPlateDataset(Dataset):
    """plate_train_dataset
    """
    def __init__(self, data_dict, img_size=64, transform=None):
        self.data_dict = data_dict
        self.data = self.data_dict['data']
        self.aoa = self.data_dict['aoa']
        self.wake_dis = self.data_dict['wake_dis']
        self.drag_coefficient = self.data_dict['drag_coefficient']
        self.case_name = self.data_dict['case_name']
        self.frequency = self.data_dict['frequency']
        self.theory_cd = self.data_dict['theory_cd']

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        wake_dis = self.wake_dis[index]
        aoa = self.aoa[index]
        drag_coefficient = self.drag_coefficient[index]
        case_name = self.case_name[index]
        frequency = self.frequency[index]
        theory_cd = self.theory_cd[index]

        if self.transform:
            data = self.transform(data)
        return data, wake_dis, aoa, drag_coefficient, case_name, theory_cd, frequency
"""
