# -*- coding: utf-8 -*-
# Author: Lx
# Date: 2020/1/4 10:53
# 通用工具

import os
import json
import time
import joblib
import base64
import numpy as np
import pandas as pd
import pypinyin
import shelve
import networkx as nx
from datetime import datetime
from functools import wraps


def dict_list_order_reverse(input_list, order_key_name):
    input_list.sort(key=lambda x: x[order_key_name], reverse=True)
    return input_list


def dict_list_order(input_list, order_key_name):
    input_list.sort(key=lambda x: x[order_key_name])
    return input_list


def dict_order_reverse(input_dict):
    return sorted(input_dict.items(), key=lambda item: item[1], reverse=True)


def dict_order(input_dict):
    return sorted(input_dict.items(), key=lambda item: item[1])


def get_key(input_dict, value):
    return [k for k, v in input_dict.items() if v == value]


def local_time_format():
    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return local_time


def now_string():
    now_time_string = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    return now_time_string


def local_date_format():
    local_date = time.strftime('%Y%m%d', time.localtime(time.time()))
    return local_date


def open_object_general(data_dir, dict_name):
    """打开json文件"""
    with open(os.path.join(data_dir, dict_name + '.json'), 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


def second_element(elem):
    return elem[1]


def df2excel(df, output_dir, file_name):
    writer = pd.ExcelWriter(os.path.join(output_dir, file_name + '.xlsx'))
    df.to_excel(writer)
    writer.save()


def save_object_general(event_graph_dict, data_dir, dict_name):
    with open(os.path.join(data_dir, dict_name + '.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(event_graph_dict, sort_keys=True, indent=2, cls=NpEncoder,
                           separators=(',', ': '), ensure_ascii=False))


def save_object_general_1(event_graph_dict, data_dir, dict_name):
    with open(os.path.join(data_dir, dict_name + '.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(event_graph_dict, sort_keys=True, indent=2))


def dict_union(ori_dict, nested_dict_list):
    """执行多个字典的合并"""
    for nested_dict in nested_dict_list:
        ori_dict = dict(ori_dict, **nested_dict)
    return ori_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def concat_run_string(rule_name, ele_quantity):
    No_list = ['No' + str(i) for i in range(ele_quantity)]
    sentence = ''
    p_str = ''
    for i in range(len(No_list)):
        p_str += '{} '
        No = No_list[i]
        if i != (len(No_list) - 1):
            sentence += 'c.' + No + '.id, '
        else:
            sentence += 'c.' + No + '.id'

    concat_string = "print('{} detected -> {}'.format({}))".format(rule_name, p_str, sentence)
    return concat_string


def get_groupBy_list(input_df, groupBy_cols, group_sub_list_key):
    """
    :param input_df:
    :param groupBy_cols:
    :param group_sub_list_key:
    :return: 返回大厦大厦大厦
    """
    group_result_list = list(input_df.groupby(groupBy_cols))
    ret = []  # 函数返回形式为列表
    for i in range(len(group_result_list)):
        group_tuple = group_result_list[i]
        group_dict = {}
        group_value = group_tuple[0]  # 即分组字段的某一个值
        group_df = group_tuple[1]  # 相当于对分组列进行筛选后，每个分组列值对应的DataFrame.

        for j in range(len(groupBy_cols)):
            col = groupBy_cols[j]
            if len(groupBy_cols) > 1:
                group_dict[col] = group_value[j]
            else:
                group_dict[col] = group_value
        group_sub_list = group_df.to_dict(orient='records')
        # 对于group_sub_list，其中的每个字典的key又包含了groupBy_cols中的key，可视情况去除
        for sub_dict in group_sub_list:
            for col in groupBy_cols:
                sub_dict.pop(col)
        group_dict[group_sub_list_key] = group_sub_list
        ret.append(group_dict)
    return ret


def get_time_diff_expr(diff_minutes):
    """
    :param diff_minutes: 两个时间的分钟整数差
    :return: 转化为X年X月X日 X时X分的中文描述
    """
    h, m = divmod(diff_minutes, 60)
    d, h = divmod(h, 24)
    M, d = divmod(d, 30)
    Y, M = divmod(M, 12)
    time_diff_expr = '{}年{}月{}日 {}时{}分'.format(Y, M, d, h, m)
    return time_diff_expr


def flatten(li):
    """
    将子列表元素展开
    :param li:
    :return:
    """
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])


def add_element_alternately(input_list, list_output=None, x=0):
    """
    # 交替插入input_list中每个子列表中的元素
    :param input_list:  待进行元素交替保存的原始嵌套列表
    :param x: 子列表索引，默认从0开始
    :param list_output: 输出的结果列表
    :return:
    """
    # print(x)
    if list_output is None:
        list_output = []
    for sub_list in input_list:
        if x >= len(sub_list):
            continue
        else:
            list_output.append(sub_list[x])
    x += 1
    if len(list_output) < len(flatten(input_list)):
        return add_element_alternately(input_list, list_output, x)
    else:
        return list_output


def add_element_alternately_new(input_list):
    """
    # 交替插入input_list中每个子列表中的元素
    :param input_list:  待进行元素交替保存的原始嵌套列表
    :return:
    """
    # print(x)
    output_list = []
    max_len = max([len(sub_list) for sub_list in input_list])
    for i in range(max_len):
        for sub_list in input_list:
            if len(sub_list) <= i:
                continue
            else:
                output_list.append(sub_list[i])
    return list_drop_dup_keeping_order(output_list)


def list_drop_dup_keeping_order(input_list):
    """
    :param input_list: 输入的列表
    :return: 在保留input_list的原始元素顺序时，对input_list进行去重
    """
    output_list = list(set(input_list))
    output_list.sort(key=input_list.index)
    return output_list


def minus_float_pow_check(base_num, exponent):
    """
    对输入的基数进行判断，若base_num为负，返回-abs(base_num)**exponent
    :param base_num: 基数
    :param exponent: 次幂
    :return:
    """
    if base_num < 0:
        return -abs(base_num) ** exponent
    else:
        return base_num ** exponent


class GetFileNameFromPath(object):
    """
    用于处理文件路径的信息分解
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def file_name_with_suffix(self):
        return os.path.basename(self.file_path)

    def file_name_without_suffix(self):
        name_with_suffix = os.path.basename(self.file_path)
        name_without_suffix = name_with_suffix.split('.')[0]
        return name_without_suffix

    def file_path_without_suffix(self):
        return os.path.splitext(self.file_path)[0]

    def only_suffix(self):
        return os.path.splitext(self.file_path)[-1]


class GetInfoFromFileName(object):
    """
    用于处理文件名的信息分解
    """

    def __init__(self, file_name):
        self.file_name = file_name

    def get_filename_suffix(self):
        suffix = os.path.splitext(self.file_name)[-1]
        return suffix

    def get_filename_without_suffix(self):
        name_without_suffix = os.path.splitext(self.file_name)[0]
        return name_without_suffix


def get_time_diff(df, col_time):
    """
    用所有数据的当前时间减去上一次时间
    :return:
    """
    df[col_time] = pd.to_datetime(df[col_time])
    time_series = df[col_time]
    choices = time_series.shift(0) - time_series.shift(1)
    choices_new = [item.seconds for item in list(choices) if item is not None]
    conditions = [True]
    df['time_diff'] = np.select(choicelist=[choices_new], condlist=conditions, default=0)
    df['time_diff'].fillna(0, inplace=True)
    return df


def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


def obj_save(output_dir, file_name, **kwargs):
    obj_file = shelve.open(os.path.join(output_dir, file_name))
    for key, value in kwargs.items():
        obj_file[key] = value
    obj_file.close()


def obj_open(input_dir, file_name):
    obj_file = shelve.open(os.path.join(input_dir, file_name))
    return obj_file


def progress_print(func_name):
    print(func_name + ' completed!')


def to_excel_with_mul_sheets(output_dir, save_name, **kwargs):
    with pd.ExcelWriter(os.path.join(output_dir, save_name)) as writer:
        for sheet, df in kwargs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
        writer.close()


def graph_save(graph, save_dir, file_name):
    save_dict = dict(nodes=[[n, graph.nodes[n]] for n in graph.nodes()],
                     edges=[[u, v, graph.edges[(u, v)]] for u, v in graph.edges()])
    save_object_general(save_dict, save_dir, file_name)


def open_graph(open_dir, fname):
    G = nx.Graph()
    json_file = open_object_general(open_dir, fname)
    G.add_nodes_from(json_file['nodes'])
    G.add_edges_from(json_file['edges'])
    return G


def make_dir_if_not_exists(input_dir):
    """
    判断一个文件夹是否存在，如不存在，则创建该文件夹
    :return:
    """
    if os.path.exists(input_dir):
        pass
    else:
        os.makedirs(input_dir)


def read_json(path):
    with open(path, 'r', encoding='utf-8')as f:
        json_data = json.load(f)
        return json_data


def write_json(info_list, path):
    with open(path, 'w', encoding='utf-8')as f:
        f.write(json.dumps(info_list, indent=2, ensure_ascii=False))
        return 'save!'


def make_dir_if_not_exists_and_del(f_input_dir, s_input_dir):
    """
    每次只保留新的目录和文件   删除其他所有旧的
    父目录  f_input_dir
    子目录  s_input_dir
    删除当前文件下的所有文件夹和文件
    :return:
    """
    # 删除
    for root, dirs, files in os.walk(f_input_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    # 在创建
    os.makedirs(s_input_dir)
    return


def make_dir_exists(input_dir):
    """
    判断一个文件夹是否存在，如不存在，则创建该文件夹
    :return:
    """
    if os.path.exists(input_dir):
        return 1
    os.makedirs(input_dir)
    return 0


def judge_dir_exists(input_dir):
    """
    判断一个文件夹是否存在，如不存在，返回0
    :return:
    """
    if os.path.exists(input_dir):
        return 1
    return 0


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer


def save_txt(data, save_dir, name):
    with open(os.path.join(save_dir, name + '.txt'), 'w', encoding='utf-8') as f:
        f.write(data)


def min_absolute_error(real_y, pred_y):
    """
    计算最小绝对误差，其中入参的两个序列（列表），长度应一致
    :param real_y: 真实值序列
    :param pred_y: 预测值序列
    :return:
    """
    abs_error_list = []
    for i in range(len(real_y)):
        real_item = real_y[i]
        pred_item = pred_y[i]
        abs_error = abs(pred_item - real_item)
        abs_error_list.append(abs_error)

    return min(abs_error_list), abs_error_list.index(min(abs_error_list))


def max_absolute_error(real_y, pred_y):
    """
    计算最大绝对误差，其中入参的两个序列（列表），长度应一致
    :param real_y: 真实值序列
    :param pred_y: 预测值序列
    :return:
    """
    abs_error_list = []
    for i in range(len(real_y)):
        real_item = real_y[i]
        pred_item = pred_y[i]
        abs_error = abs(pred_item - real_item)
        abs_error_list.append(abs_error)

    return max(abs_error_list), abs_error_list.index(max(abs_error_list))


def save_string_to_txt(string, filename):
    with open(filename, 'w', encoding='UTF-8') as file:
        file.write(string)


def print_num_params(model):
    """计算大模型参数量"""
    num_params = sum(p.numel() for p in model.parameters())
    return num_params


def image2base64(file_path):
    with open(file_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def local_pdf_to_base64(file_path: str) -> str:
    """
    将本地 PDF 文件转换为 Base64 字符串
    :param file_path: PDF 文件本地路径（例如："./documents/sample.pdf"）
    :return: Base64 编码字符串
    """
    try:
        # 以二进制模式读取文件
        with open(file_path, "rb") as pdf_file:
            # 读取文件内容并编码为 Base64
            base64_data = base64.b64encode(pdf_file.read())
            # 将字节类型转换为 UTF-8 字符串
            return base64_data.decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在：{file_path}")
    except Exception as e:
        raise RuntimeError(f"转换失败：{str(e)}")


def drop_thinking_strings(input_string):
    think_end_index = input_string.index("</think>")
    answer_start_index = think_end_index + 8
    return input_string[answer_start_index:]


def deduplicate_documents_in_hash(documents):
    """基于文档内容哈希的去重方法"""
    seen = set()
    unique_documents = []

    for doc in documents:
        # 创建文档内容的哈希值
        content_hash = hash(doc.page_content.strip())

        if content_hash not in seen:
            seen.add(content_hash)
            unique_documents.append(doc)

    return unique_documents


def clean_input(user_input: str) -> str:
    """去除字符串首尾的冗余引号（单引号或双引号）"""
    if len(user_input) >= 2:
        # 检查首尾是否匹配的同类型引号
        if (user_input.startswith('"') and user_input.endswith('"')) or \
           (user_input.startswith("'") and user_input.endswith("'")):
            return user_input[1:-1]  # 切片去除首尾字符
    return user_input  # 无需处理则返回原字符串


def get_wanted_tool_call(tool_calls: list, names: set) -> dict:
    """从AIMessage中的tool_calls的属性中，根据name，获取希望获取的tool_call"""
    wanted_tool_call = dict()
    for tool_call in tool_calls:
        if tool_call['name'] in names:
            wanted_tool_call[tool_call['name']] = tool_call
            if len(wanted_tool_call) == len(names):
                break
        else:
            continue
    return wanted_tool_call
