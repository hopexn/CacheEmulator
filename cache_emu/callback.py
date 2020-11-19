class Callback:
    def __init__(self, interval: int = 1, test_mode=False, **kwargs):
        self.interval = interval
        self.test_mode = test_mode
        
        self.i_step = 0
        self.i_episode = 0
    
    def on_step_end(self):
        self.i_step += 1
        if self.i_step % self.interval == 0:
            self.i_episode += 1
            return self.on_episode_end()
    
    def on_episode_end(self):
        pass
    
    def on_game_begin(self):
        pass
    
    def on_game_end(self):
        pass
    
    def switch_mode(self, test_mode=False):
        self.test_mode = test_mode
    
    def reset(self):
        self.i_step = 0
        self.i_episode = 0


class CallbackManager:
    def __init__(self):
        self.callbacks = []
    
    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_step_end(self):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_step_end()
            if res is not None:
                info.update(res)
        
        return info
    
    def on_game_begin(self):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_game_begin()
            if res is not None:
                info.update(res)
        
        return info
    
    def on_game_end(self):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_game_end()
            if res is not None:
                info.update(res)
        
        return info
    
    def switch_mode(self, test_mode=False):
        for cb in self.callbacks:
            cb.switch_mode(test_mode)
        print("Switch to {} mode.".format("test" if test_mode else "train"))
    
    def reset(self):
        for cb in self.callbacks:
            cb.reset()
