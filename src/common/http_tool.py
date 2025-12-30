# -*- coding: utf-8 -*-
"""
@Time ： 2022/7/28 14:13
@Auth ： luanxing
@File ：http_tool.py
@IDE ：PyCharm
HTTP连接常用工具
"""

import requests


def request_post(url, param):
    """
    :param url: http地址
    :param param: body体, json格式
    :return:
    """
    try:
        headers = {"Content-Type": "application/json; charset=utf-8",
                   "User-Agent": "Mozilla/5.0.html (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/39.0.html.2171.71 Safari/537.36"}
        response = requests.post(url=url, data=param, headers=headers)
        return response.json()
    except Exception as e:
        print(e)
