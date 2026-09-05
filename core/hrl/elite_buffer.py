#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elite Experience Buffer — 高质量 episode 经验优先存储
"""
import random


class EliteBuffer:
    """
    按 episode 总奖励排序的精英经验池。
    高奖励 episode 的 transition 被优先采样，引导策略复现成功轨迹。
    """

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []   # List[(transitions, episode_reward)]

    def add_episode(self, transitions, episode_reward):
        if not transitions:
            return
        self.buffer.append((list(transitions), float(episode_reward)))
        
        self.buffer.sort(key=lambda x: x[1], reverse=True)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[:self.capacity]

    def sample(self, batch_size):
        """从精英 episode 中随机采样 transition（每个 episode 取一条）"""
        if not self.buffer:
            return []
        n_eps = min(batch_size, len(self.buffer))
        selected = random.sample(self.buffer, n_eps)
        transitions = []
        for eps_transitions, _ in selected:
            if eps_transitions:
                transitions.append(random.choice(eps_transitions))
        return transitions

    def clear(self):
        """清空 buffer，供 clear_replay_buffers() 调用。"""
        self.buffer.clear()

    def __len__(self):
        return sum(len(t) for t, _ in self.buffer)

    @property
    def n_episodes(self):
        return len(self.buffer)