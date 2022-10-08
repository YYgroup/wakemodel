# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:06:13 2021
comapre the force of CNN and DNS in the validation set

@author: Wenwen Tong, Peking University
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

def cal_err(force_pre, force_dns, positions):
    positions = np.array(positions)
    force_pre = np.array(force_pre)
    force_dns = np.array(force_dns)
    error = (force_pre - force_dns)/force_dns
#    error = np.absolute(error)[positions < 10.01]
    error = np.absolute(error)[(positions < 15) & (positions > 5)]

    mean_error = np.mean(error)
    return mean_error


def draw_result(case_result, case_path, fig_name_postfix = 'case'):
    gt_wake_dis = case_result['gt_wake_dis']
    pre_drag_coefficient = case_result['pre_drag_coefficient']
    gt_drag_coefficient = case_result['gt_drag_coefficient']
    order = np.argsort(gt_wake_dis)
    gt_wake_dis = gt_wake_dis[order]
    pre_drag_coefficient = pre_drag_coefficient[order]
    
    plt.figure()
    plt.plot(gt_wake_dis, gt_drag_coefficient, color='k', label='DNS',linestyle=':')
    
    bar_height = gt_drag_coefficient*0.2
    plt.barh(gt_drag_coefficient, 15, height=bar_height, color='silver')

    plt.scatter(gt_wake_dis, pre_drag_coefficient, color='r', label ='Prediction',zorder=2)
    
    plt.xlabel('x')
    plt.ylabel(r'$C_d$')
    plt.xlim(5, 15)
    plt.xticks([5, 10, 15])
    plt.legend()
    save_path = os.path.join(case_path,'fig_force_'+fig_name_postfix+'.png')
    plt.savefig(save_path, bbox_inches='tight')

root_dir = '../experiment/stationary_plate'


for case_name in sorted(os.listdir(root_dir)):
    print('')
    print('case_name:', case_name)
    case_path = os.path.join(root_dir, case_name)
    file_name = 'test_loader_data_25.npz'

    file_path = os.path.join(case_path, file_name)
    data = np.load(file_path, allow_pickle=True)

    pre_drag_coefficient = data['pre_drag']
    gt_drag_coefficient = data['gt_drag']
    pre_wake_dis = data['pre_wake_dis']*10.0
    gt_wake_dis = data['gt_wake_dis']*10.0
    gt_case_name = data['gt_case_name']

    n = len(gt_case_name)
    case_names = np.unique(gt_case_name)
    #print('case_names:', case_names)

    res = {}  # save case in the dict
    for i in range(len(case_names)):
        cur_case = case_names[i]
        res[cur_case] = {}
        res[cur_case]['pre_drag_coefficient'] = pre_drag_coefficient[gt_case_name == cur_case]
        res[cur_case]['gt_drag_coefficient'] = gt_drag_coefficient[gt_case_name == cur_case]
        res[cur_case]['gt_wake_dis'] = gt_wake_dis[gt_case_name == cur_case]
    mean_errors= []
    for case in  res:
        pre_drag_coefficient = res[case]['pre_drag_coefficient']
        gt_drag_coefficient = res[case]['gt_drag_coefficient']
        gt_wake_dis = res[case]['gt_wake_dis']
        order = np.argsort(gt_wake_dis)
        gt_wake_dis= gt_wake_dis[order]
        pre_drag_coefficient = pre_drag_coefficient[order]

        mean_error = cal_err(pre_drag_coefficient, gt_drag_coefficient, gt_wake_dis)
        mean_errors.append(mean_error)
            
        print('case:', case)
        print('mean error:', mean_error)
#         draw_result(res[case], case_path, fig_name_postfix = case)
    print('total mean error:', np.mean(mean_errors))

