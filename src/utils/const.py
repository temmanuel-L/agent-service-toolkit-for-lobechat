from dataclasses import dataclass
from typing import List


class PressureLevel:
    # 超高压燃气管道A级：压力为2.5<P≤4.0MPa
    high_A = [2.5e6, 4.0e6]
    # 高压燃气管道B级：压力为1.6<P≤2.5MPa
    high_B = [1.6e6, 2.5e6]
    # 次高压燃气管道A级：压力为0.8<P≤1.6MPa
    sub_high_A = [0.8e6, 1.6e6]
    # 次高压燃气管道B级：压力为0.4<P≤0.8MPa
    sub_high_B = [0.4e6, 0.8e6]
    # 中压燃气管道A级：压力为0.2<P≤0.4MPa
    medium_A = [0.2e6, 0.4e6]
    # 中压燃气管道B级：压力为0.01≤P≤0.2MPa
    medium_B = [0.01e6, 0.2e6]
    # 低压燃气管道：压力为P<0.01MPa
    low = [0, 0.01e6]

class SimulateToolConst:
    # 管道下游用户追踪
    ignore_flow = 1e-3  # 忽略的流量，此流量的管道、用户不进行计算以及统计
    
    # 管网参数一般性检查
    normal_inner_d = [0.015, 1.0]      # 正常的内部管道直径范围,单位m
    normal_roughness = [1e-5, 1e-4]    # 正常的管道粗糙度范围,单位m
    normal_pipe_length = [0, 10000.0]  # 正常管道长度范围,单位m
    
    # 压力级别设置，表压，单位Pa
    pressure_level = PressureLevel
def get_pressure_level(pressure: float) -> str:
    """
    根据压力值获取压力级别
    """
    for level_name in dir(SimulateToolConst.pressure_level):
        if not level_name.startswith('_'):
            level_range = getattr(SimulateToolConst.pressure_level, level_name)
            p_min, p_max = level_range
            if p_min < pressure <= p_max:
                return level_name
    return "unknown"

