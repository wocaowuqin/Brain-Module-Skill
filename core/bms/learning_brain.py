from __future__ import annotations
import random
from collections import deque
from .base import BMSContext, BaseBrain

class BrainSafetyGuard:
    def __init__(self, patience=3): self.patience=int(patience); self.last=None; self.count=0
    def apply(self, action, reward=0.0):
        if action == self.last and float(reward) <= 0: self.count += 1
        else: self.last, self.count = action, 0
        return "noop" if self.count >= self.patience else action

class LearningBrainAgent(BaseBrain):
    MACRO_ACTIONS = ("noop", "call_deployment", "call_migration", "call_reroute")
    def __init__(self, model=None, rule_brain=None, epsilon=0.1, seed=None):
        self.model=model; self.rule_brain=rule_brain; self.epsilon=float(epsilon); self.rng=random.Random(seed); self.guard=BrainSafetyGuard()
    def _encode_state(self, context: BMSContext):
        loads=list(context.metrics.get("loads", [])); lifetimes=[float(x.get("leave_time",0))-context.timestamp for x in context.active_requests]
        q=sum(float(v) for v in context.queue_backlog.values()); freq=context.metrics.get("module_call_frequency", {})
        return [float(sum(loads)/len(loads) if loads else 0), float(sum((x-(sum(lifetimes)/len(lifetimes)))**2 for x in lifetimes)/len(lifetimes) if lifetimes else 0), q, *[float(freq.get(a,0)) for a in self.MACRO_ACTIONS]]
    def _get_instruction(self, action_type, context):
        return {"action_type": action_type, "target": context.active_requests[0].get("id") if action_type == "call_migration" and context.active_requests else None}
    def decide(self, context):
        if self.rng.random() < self.epsilon and self.rule_brain:
            action = self.rule_brain.decide(context).get("action_type", "noop")
        elif self.model is not None:
            action_id, _ = self.model.predict(self._encode_state(context), deterministic=True); action=self.MACRO_ACTIONS[int(action_id)]
        else: action="noop"
        action=self.guard.apply(action)
        return {**self._get_instruction(action, context), "reasoning": "learning brain macro-action selection"}
    def arbitrate(self, module_result, context):
        return module_result if module_result.get("confidence", 0.0) >= 0.0 else {"action_type":"noop","reasoning":"rejected low confidence"}
