
import numpy as np

class WorldlineSwitchDetector:
    """世界线切换检测 — Strubbe框架的"偏离平衡态重新打开世界线空间"
    
    维护滑动窗口追踪最近N次观测的最大似然值。
    当窗口满且所有似然 < 阈值 → 环境已切换 → 重置世界线概率。
    """
    
    def __init__(self, n_worldlines=3, window_size=5, threshold=0.05):
        self.n_wl = n_worldlines
        self.window_size = window_size
        self.threshold = threshold
        self.recent_likelihoods = []  # 最大似然滑动窗口
        self.switch_count = 0
        self.total_checks = 0
    
    def check_and_reset(self, worldline_probs, log_likelihoods):
        """返回 (是否触发重置, 新worldline_probs)"""
        self.total_checks += 1
        
        # 记录本次最大似然
        max_ll = float(max(log_likelihoods)) if len(log_likelihoods) > 0 else 0.0
        self.recent_likelihoods.append(max_ll)
        if len(self.recent_likelihoods) > self.window_size:
            self.recent_likelihoods.pop(0)
        
        # 检测条件: 窗口满 且 所有似然 < 阈值
        if len(self.recent_likelihoods) >= self.window_size:
            if all(ll < self.threshold for ll in self.recent_likelihoods):
                self.switch_count += 1
                self.recent_likelihoods = []
                new_probs = np.ones(self.n_wl) / self.n_wl
                return True, new_probs
        
        return False, worldline_probs
    
    def get_stats(self):
        return {
            'switch_count': self.switch_count,
            'total_checks': self.total_checks,
            'switch_rate': self.switch_count / max(self.total_checks, 1),
        }
