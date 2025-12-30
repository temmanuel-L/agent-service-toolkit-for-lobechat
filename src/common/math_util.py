# -*- coding: utf-8 -*-
# Author: Lx
# Date: 2021/5/16 16:13


def get_trough(input_list):
    """
    :param input_list: 输入的数值列表
    :return: 获取数值序列的波谷值
    """
    low_seq = []
    for i in range(0, len(input_list) - 1):
        if i == 0 and input_list[i] < input_list[i + 1]:    # 头部数据判断
            low_seq.append(input_list[i])
        else:
            if (input_list[i - 1] > input_list[i] and   # 中间数据判断
                    input_list[i + 1] > input_list[i]):
                low_seq.append(input_list[i])
    if input_list[-1] < input_list[-2]:     # 尾部数据判断
        low_seq.append(input_list[-1])
    return low_seq


def percentile_value(input_list, input_p):
    """
    :param input_list: 输入的数值序列
    :param input_p: 需要的分位数
    :return:
    """
    input_list.sort(reverse=True)   # 将将输入列表数据按照由大到小排列
    # 接下来需要获取按照分位数值生成的列表元素序号
    seq_num = input_p * len(input_list)
    # 该序号可能为小数，需四舍五入为整数，但若结果为0，需要+1
    seq_num_int = int(round(round(seq_num * 100) / 100.0, 0))
    if seq_num_int == 0:
        seq_num_int += 1
    # 对于input_list来说，其元素序号应该比seq_num_int减1
    seq_num_final = seq_num_int - 1
    return input_list[seq_num_final]

