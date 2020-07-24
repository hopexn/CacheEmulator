import ctypes

import numpy as np

from utils import proj_utils, ctypes_utils, bert_utils
from utils.mpi_utils import sync_time_span

lib_path = "src/cache_emu/build/libcacheemu.so"
lib_cache_emu = ctypes_utils.load_lib(proj_utils.abs_path(lib_path))

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


def init_loader(data, interval=1):
    print("data:\n", data.head())
    
    content_ids = np.array(data['content_id'], dtype=np.int32)
    timestamps = np.array(data['timestamp'], dtype=np.int32)
    
    num_requests = len(content_ids)
    lib_cache_emu.load_dataset(content_ids.ctypes, timestamps.ctypes, num_requests)
    
    t_beg, t_end = sync_time_span(timestamps[0], timestamps[-1])
    num_steps = lib_cache_emu.slice_dataset_by_time(int(t_beg), int(t_end), interval)
    
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
        
        if use_bert_feature:
            # 由于bert_utils需要feature_dims()，因此bert_utils的初始化需要放到最后
            bert_utils.init(out_feature_dim=self.feature_dims())
    
    def get_features(self, contents: np.array):
        assert (contents.dtype == np.int32)
        num_contents = contents.shape[0]
        feature_struct = lib_cache_emu.get_features(self.handler, contents.ctypes, num_contents)
        features = ctypes_utils.buffer_to_numpy(feature_struct, np.float32)
        features = features.reshape((num_contents, self.feature_dims()))
        
        alpha = 0.5
        if bert_utils.enabled:
            bert_features = bert_utils.forward(contents)
            bert_utils.backward(contents, features)
            features = features * alpha + bert_features * (1 - alpha)
        
        return features
    
    def finished(self):
        return bool(lib_cache_emu.finished(self.handler))
    
    def get_mean_hit_rate(self):
        return float(lib_cache_emu.get_mean_hit_rate(self.handler))
    
    def get_i_episode(self):
        return lib_cache_emu.get_i_episode(self.handler)
    
    def on_episode_end(self):
        return lib_cache_emu.on_episode_end(self.handler)
