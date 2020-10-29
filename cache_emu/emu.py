import ctypes
import os

import numpy as np

from .utils import ctypes_utils, proj_utils

# 项目根目录
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

lib_path = proj_utils.abs_path("./build/libcacheemu.so")
lib_cache_emu = ctypes_utils.load_lib(lib_path)

ctypes_utils.setup_res_type(lib_cache_emu.load_dataset, ctypes.c_void_p)
ctypes_utils.setup_res_type(lib_cache_emu.slice_dataset_by_time, ctypes.c_int32)
ctypes_utils.setup_res_type(lib_cache_emu.init_cache_emu, ctypes.c_int32)
ctypes_utils.setup_res_type(lib_cache_emu.step, ctypes_utils.Triple)
ctypes_utils.setup_res_type(lib_cache_emu.get_cache_contents, ctypes_utils.IntBuffer)
ctypes_utils.setup_res_type(lib_cache_emu.get_candidates, ctypes_utils.IntBuffer)
ctypes_utils.setup_res_type(lib_cache_emu.get_candidate_frequencies, ctypes_utils.FloatBuffer)
ctypes_utils.setup_res_type(lib_cache_emu.get_step_elements, ctypes_utils.IntBuffer)
ctypes_utils.setup_res_type(lib_cache_emu.feature_dims, ctypes.c_size_t)
ctypes_utils.setup_res_type(lib_cache_emu.setup_traditional_feature_types, ctypes.c_void_p)
ctypes_utils.setup_res_type(lib_cache_emu.setup_swlfu_feature_types, ctypes.c_void_p)
ctypes_utils.setup_res_type(lib_cache_emu.get_features, ctypes_utils.FloatBuffer)
ctypes_utils.setup_res_type(lib_cache_emu.get_mean_hit_rate, ctypes.c_float)
ctypes_utils.setup_res_type(lib_cache_emu.finished, ctypes.c_int32)
ctypes_utils.setup_res_type(lib_cache_emu.on_episode_end, ctypes.c_float)


def init_loader(data, t_beg: int, t_end: int, t_interval=1):
    print("data:\n", data.head())
    
    content_ids = np.array(data['content_id'], dtype=np.int32)
    timestamps = np.array(data['timestamp'], dtype=np.int32)
    
    num_requests = len(content_ids)
    lib_cache_emu.load_dataset(content_ids.ctypes, timestamps.ctypes, num_requests)
    
    num_steps = lib_cache_emu.slice_dataset_by_time(int(t_beg), int(t_end), t_interval)
    
    return num_requests, num_steps, (t_beg, t_end)


class CacheEmu:
    def __init__(self, capacity, passive_mode=False):
        self.capacity = capacity
        
        self.handler = lib_cache_emu.init_cache_emu(capacity, passive_mode)
        self.last_contents = None
    
    def reset(self):
        lib_cache_emu.reset(self.handler)
    
    def step(self):
        return lib_cache_emu.step(self.handler)
    
    def get_step_elements(self):
        res = lib_cache_emu.get_step_elements(self.handler)
        return ctypes_utils.buffer_to_numpy(res, np.int32)
    
    def get_candidates(self):
        res = lib_cache_emu.get_candidates(self.handler)
        return ctypes_utils.buffer_to_numpy(res, np.int32)
    
    def get_cache_contents(self):
        res = lib_cache_emu.get_cache_contents(self.handler)
        return ctypes_utils.buffer_to_numpy(res, np.int32)
    
    def get_cache_content_frequencies(self):
        res = lib_cache_emu.get_cache_content_frequencies(self.handler)
        return ctypes_utils.buffer_to_numpy(res, np.float32)
    
    def get_candidate_frequencies(self):
        res = lib_cache_emu.get_candidate_frequencies(self.handler)
        return ctypes_utils.buffer_to_numpy(res, np.float32)
    
    def update_cache(self, new_contents: np.array):
        assert (new_contents.dtype == np.int32)
        lib_cache_emu.update_cache(self.handler, new_contents.ctypes, new_contents.shape[0])
    
    def feature_dims(self):
        return lib_cache_emu.feature_dims(self.handler)
    
    def setup_features(self, use_lfu_feature: bool = False, use_lru_feature: bool = False,
                       use_ogd_opt_feature: bool = False,
                       use_bert_feature=False,
                       wlfu_w_lens: list = [], **kwargs):
        lib_cache_emu.setup_traditional_feature_types(
            self.handler,
            use_lfu_feature,
            use_lru_feature,
            use_ogd_opt_feature
        )
        
        wlfu_w_lens = np.array(wlfu_w_lens, dtype=np.int32)
        lib_cache_emu.setup_swlfu_feature_types(self.handler, wlfu_w_lens.ctypes, wlfu_w_lens.shape[0])
    
    def get_features(self, contents: np.array):
        assert (contents.dtype == np.int32)
        num_contents = contents.shape[0]
        feature_struct = lib_cache_emu.get_features(self.handler, contents.ctypes, num_contents)
        features = ctypes_utils.buffer_to_numpy(feature_struct, np.float32)
        features = features.reshape((num_contents, self.feature_dims()))
        
        return features
    
    def finished(self):
        return bool(lib_cache_emu.finished(self.handler))
    
    def get_mean_hit_rate(self):
        return float(lib_cache_emu.get_mean_hit_rate(self.handler))
    
    def get_i_episode(self):
        return lib_cache_emu.get_i_episode(self.handler)
    
    def on_episode_end(self):
        return lib_cache_emu.on_episode_end(self.handler)
