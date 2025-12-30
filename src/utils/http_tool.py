"""
@Time ： 2022/7/28 14:13
@Auth ： luanxing
@File ：http_tool.py
@IDE ：PyCharm
HTTP连接常用工具
"""

# --coding: utf-8--**
import requests

from utils.log_utils import get_logger

logger = get_logger(__name__)


def request_post(url, param):
    """
    :param url: http地址
    :param param: body体, json格式
    :return:
    """
    try:
        # headers = {"Content-Type": "application/json; charset=utf-8"}
        headers = {"charset": "utf-8", "application": "json"}
        response = requests.post(url=url, json=param, headers=headers)
        return response.json()
    except Exception as e:
        logger.error(f"稳态服务访问失败: {e}")
        raise ConnectionError(f"稳态服务访问失败: {e}")
