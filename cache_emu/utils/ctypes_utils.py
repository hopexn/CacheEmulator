import ctypes

import numpy as np


# 加载动态链接库
def load_lib(lib_path):
    return ctypes.cdll.LoadLibrary(lib_path)


# 设置返回类型
def setup_res_type(func, c_type):
    func.restype = c_type


# 设置参数类型
def setup_arg_types(func, c_types):
    func.argtypes = c_types


# 用于传递int32类型数组
class IntBuffer(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_int32)),
        ('size', ctypes.c_size_t)
    ]


# 用于传递float32类型数组
class FloatBuffer(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('size', ctypes.c_size_t)
    ]


# 用于传递三元组
class Triple(ctypes.Structure):
    _fields_ = [
        ('first', ctypes.c_size_t),
        ('second', ctypes.c_size_t),
        ('third', ctypes.c_size_t)
    ]
    
    def tuple(self):
        return self.first, self.second, self.third


# 将返回的buffer转成对应的numpy数组
def buffer_to_numpy(buf, dtype: np.dtype):
    # 获取对应的C类型
    c_type = np.ctypeslib.as_ctypes_type(dtype)
    data_np = np.zeros(buf.size, dtype=dtype)
    ctypes.memmove(data_np.ctypes, buf.data, ctypes.sizeof(c_type) * buf.size)
    return data_np
