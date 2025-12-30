# -*- coding: utf-8 -*-
"""
@Time ： 2025/12/9 17:14
@Auth ： luanxing
@File ：beautify_json.py
@IDE ：PyCharm
"""

from common.common_tool import open_object_general, save_object_general


ori_json = open_object_general('./', 'fastapi_service_info')
new_json = save_object_general(ori_json, './', 'fastapi_service_info')