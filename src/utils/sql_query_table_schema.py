# 此模块下维护的sql查询/插入语句以及相关函数

from psycopg2 import sql
import polars as pl


class calc_res_sql_schema:
    """
    仿真结果查询语句及schema定义
    维护calc_res相关的sql语句以及查询结果schema定义
    """
    res_schema =  {
        'area_no': pl.Int32,  # 测算结果所在片区号
        'node_code': pl.Utf8,  # 元件代码
        'source_code': pl.Utf8,  # 开始节点id
        'target_code': pl.Utf8,  # 结束节点id
        'simulate_node_type': pl.Utf8,  # 仿真节点类型,regulator,customer,node,source,valve,pipeline
        'volume_flow_rate': pl.Float64,  # 瞬时流量
        'in_pressure': pl.Float64,  # 进口压力 用户仿真结果使用pressure
        'pressure': pl.Float64,  # 出口压力
        'flow_rate': pl.Float64,  # 流速
        'pipe_flow': pl.Float64,  # 管道流向
    }
    # 根据res_schema生成pipe_res_schema
    pipe_res_schema = {
        'area_no': pl.Int32,
        'pipe_code': pl.Utf8,  # 原node_code重命名为pipe_code
        'source_code': pl.Utf8,
        'target_code': pl.Utf8,
        'flow': pl.Float64,  # 原volume_flow_rate重命名为flow
        'flow_rate': pl.Float64,  # 流速 m/s
    }
    # 添加node_res_schema定义
    node_res_schema = {
        'area_no': pl.Int32,
        'node_code': pl.Utf8,
        'node_type': pl.Utf8,  # 原simulate_node_type重命名为node_type, 增加enum：reg_source, reg_customer
        'in_pressure': pl.Float64,
        'pressure': pl.Float64
    }

 
    def read_res(self, project_code, topology_code, ts, batch_no):
        """
        读取数据库获取节点数据
        :return:
        """
        # 读取节点数据
        sql_query = sql.SQL("""
        SELECT 
            area_no, 
            node_code, 
            source_code,
            target_code, 
            simulate_node_type, 
            volume_flow_rate, 
            in_pressure, pressure, 
            flow_rate, 
            pipe_flow
        FROM public.dt_compute_result
        where project_code = %s
        and topology_code = %s
        and ts = %s
        and batch_no = %s;
        """)
        params = (project_code, topology_code, ts, batch_no)
        return sql_query, params, self.res_schema
    

class topo_sql_schema:
    """
    拓扑查询语句及schema定义
    维护topo相关的sql语句以及查询结果schema定义
    """
    # dt_node表的schema定义
    topo_node_schema = {
        'node_code': pl.Utf8,            # 节点代码（与 topologyCode 联合主键）
        'simulate_node_type': pl.Utf8,   # 仿真节点类型
        'pressure_level': pl.Utf8,       # 压力级别（更新时忽略）
    }
    
    # dt_pipeline表的schema定义
    topo_pipe_schema = {
        'pipe_code': pl.Utf8,         # 节点代码（与 topologyCode 联合主键）
        'source_code': pl.Utf8,       # 起始节点编号
        'target_code': pl.Utf8,       # 目标节点编号
        'material': pl.Utf8,          # 管材
        'length': pl.Float64,         # 管道长度
        'outer_d': pl.Float64,        # 仿真管径（SQL中simulate_diam as outer_d）
        'inner_d': pl.Float64,        # 内径（**非数据库字段**，通过外径和壁厚关系获取，具体关系需由后端提供逻辑）
        'pressure_level': pl.Utf8,    # 压力级别（更新时忽略）
        'wall_thickness': pl.Float64, # 管道壁厚
        'roughness': pl.Float64,      # 粗糙度
    }


    def read_topo_node(self, tenant_code, project_code, topology_code):
        sql_query = sql.SQL("""
        SELECT
            code as node_code,
            simulate_node_type,
            pressure_level
        FROM public.dt_node
        where tenant_code = %s
        and project_code = %s
        and topology_code = %s
        and if_simulate = true;
        """)
        params = (tenant_code, project_code, topology_code)
        return sql_query, params, self.topo_node_schema
    
    def read_topo_pipe(self, tenant_code, project_code, topology_code):
        sql_query = sql.SQL("""
        select
            code as pipe_code,
            source_code,
            target_code,
            material,
            length,
            simulate_diam as outer_d,
            pressure_level,
            wall_thickness,
            roughness
        FROM public.dt_pipeline
        where tenant_code = %s
        and project_code = %s
        and topology_code = %s
        and if_simulate = true;
        """)
        params = (tenant_code, project_code, topology_code)
        return sql_query, params, self.topo_pipe_schema


class scada_sql_schema:
    scada_schema = {
        'node_code': pl.Utf8,            # 节点代码
        'simulate_node_type': pl.Utf8,   # 仿真节点类型
        'valve_status': pl.Utf8,         # 阀门状态
        'in_pressure': pl.Float64,       # 进口压力
        'pressure': pl.Float64,          # 出口压力
        'flow': pl.Float64,              # 流量（经过/3600转换）
        'control_type': pl.Utf8,         # 控制类型
    }
    area_scada_schema = {
        'node_code': pl.Utf8,           # 节点代码
        'pressure': pl.Float64,         # 压力
        'flow': pl.Float64,             # 流量
        'simulate_node_type': pl.Utf8,  # 仿真节点类型
        'mark': pl.Utf8,                # 赋值类型
    }

    def read_scada(self, tenant_code, project_code, topology_code, ts, batch_no):
        sql_query = sql.SQL("""
        select
            node_code,
            simulate_node_type,
            valve_status,
            (in_pressure::float + 101325) as in_pressure,
            (pressure::float + 101325) as pressure,
            (flow::float / 3600.0) as flow,
            control_type
        from dt_iot_push
        where tenant_code= %s
        and project_code = %s
        and topology_code= %s
        and ts = %s
        and batch_no = %s;
        """)
        params = (tenant_code, project_code, topology_code, ts, batch_no)
        return sql_query, params, self.scada_schema
