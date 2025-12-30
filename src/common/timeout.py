# -*- coding: utf-8 -*-
# Author: Lx
# Date: 2021/4/7 11:08


from functools import wraps
import time
import os
from threading import Thread

# asdf
def set_time_limit(t):
    def auto_quit(t1):
        """此为控制进程超时退出的线程函数"""
        time.sleep(t1)
        print("time out {}".format(t1))
        # 此函数专门用于线程控制主进程退出
        os._exit(1)



    def decorator(f):
        """此函数用于传入被装饰函数f"""
        @wraps(f)
        def wrapper(*args,**kwargs):
            """装饰器内部遵循的逻辑是：
            1.auto_quit先执行完，进程结束
            2.被修饰函数f先执行完，auto_quit函数停止执行
            3.被修饰函数执行完，下面的代码才能运行
            """
            # 此处的t是set_time_limit函数的形参，是auto_quit函数的实参
            t1 = Thread(target=auto_quit,args=(t,))
            t2 = Thread(target=f,args=args,kwargs=kwargs)
            t1.setDaemon(True)  # 满足第2点
            t1.start()
            t2.start()
            t2.join()   #满足第3点
        return wrapper
    return decorator



