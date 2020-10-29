import gym
import numpy as np

from .callback import CallbackManager
from .emu import CacheEmu


class ActiveCacheEnv(gym.Env):
    def __init__(self, capacity: int, callback_manager: CallbackManager = None, feature_config={}):
        self.capacity = capacity
        
        self.emu = CacheEmu(capacity, passive_mode=False)
        self.callback_manger = callback_manager
        self.candidates = None
        
        self.emu.setup_features(**feature_config)
        
        content_dim = capacity
        feature_dim = self.emu.feature_dims()
        
        self.action_space = gym.spaces.MultiBinary(content_dim)
        self.observation_space = gym.spaces.Box(
            low=np.zeros((content_dim, feature_dim), dtype=np.float32),
            high=np.ones((content_dim, feature_dim), dtype=np.float32)
        )
    
    def reset(self):
        self.emu.reset()
        self.callback_manger.reset()
        self.callback_manger.on_game_begin()
        
        self.candidates = self.emu.get_candidates()
        observation = self.emu.get_features(self.candidates)
        return observation
    
    def close(self):
        self.callback_manger.on_game_end()
    
    def step(self, action: np.array):
        info = {}
        
        new_contents = self.candidates[action]
        self.emu.update_cache(new_contents)
        
        while True:
            n_requests_processed, n_contents_missed, n_requests_remained = self.emu.step().tuple()
            
            self.candidates = self.emu.get_candidates()
            observation = self.emu.get_features(self.candidates)
            reward = self.emu.get_candidate_frequencies()
            done = self.emu.finished()
            
            step_end_info = self.callback_manger.on_step_end()
            info.update(step_end_info)
            
            if n_requests_processed != 0:
                break
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        pass


class PassiveCacheEnv(gym.Env):
    def __init__(self, capacity: int, callback_manager: CallbackManager = None, feature_config={}):
        self.capacity = capacity
        
        self.emu = CacheEmu(capacity, passive_mode=True)
        self.callback_manger = callback_manager
        self.candidates = None
        
        self.emu.setup_features(**feature_config)
        
        content_dim = capacity + 1
        feature_dim = self.emu.feature_dims()
        
        self.action_space = gym.spaces.MultiBinary(content_dim)
        self.observation_space = gym.spaces.Box(
            low=np.zeros((content_dim, feature_dim), dtype=np.float32),
            high=np.ones((content_dim, feature_dim), dtype=np.float32)
        )
    
    def reset(self):
        self.emu.reset()
        self.callback_manger.reset()
        self.callback_manger.on_game_begin()
        
        self.candidates = np.arange(self.capacity + 1, dtype=np.int32)
        self.emu.update_cache(self.candidates[:self.capacity])
        observation = self.emu.get_features(self.candidates)
        return observation
    
    def close(self):
        self.callback_manger.on_game_end()
    
    def step(self, action: np.array):
        info = {}
        
        new_contents = self.candidates[action]
        self.emu.update_cache(new_contents)
        
        reward = np.zeros(self.capacity + 1, dtype=np.float32)
        
        while True:
            n_requests_processed, n_contents_missed, n_requests_remained = self.emu.step().tuple()
            
            reward += self.emu.get_candidate_frequencies()
            if n_requests_remained == 0:
                step_end_info = self.callback_manger.on_step_end()
                info.update(step_end_info)
            
            done = self.emu.finished()
            if n_contents_missed > 0 or done:
                break
        
        self.candidates = self.emu.get_candidates()
        observation = self.emu.get_features(self.candidates)
        done = self.emu.finished()
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        pass
