"""Brain–Module–Skill (BMS) boundary contracts."""
from .base import BMSContext, BaseBrain, BaseModule, BaseSkill, _check_layer
from .exceptions import LayerViolationError
from .registry import ModuleRegistry, SkillRegistry, register_module, register_skill
from .learning_brain import LearningBrainAgent, BrainSafetyGuard
from .msft_hirl_mapping import MSFTHIRLMappingSkill, MSFTHIRLDeploymentModule

__all__ = [
    "BMSContext", "BaseBrain", "BaseModule", "BaseSkill", "_check_layer",
    "LayerViolationError", "SkillRegistry", "ModuleRegistry",
    "register_skill", "register_module", "LearningBrainAgent", "BrainSafetyGuard",
    "MSFTHIRLMappingSkill", "MSFTHIRLDeploymentModule",
]
