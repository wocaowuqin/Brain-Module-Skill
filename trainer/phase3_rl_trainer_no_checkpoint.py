"""Phase 3 trainer variant that keeps logs/datasets but skips .pth checkpoints."""

import logging

from trainer.phase3_rl_trainer import Phase3RLTrainer


logger = logging.getLogger(__name__)


class Phase3RLTrainerNoCheckpoint(Phase3RLTrainer):
    """Drop-in Phase3 trainer that does not write checkpoint/model files."""

    def _save_checkpoint(self, episode):
        logger.info("[CheckpointDisabled] skip periodic checkpoint at ep=%s", episode)

    def _save_final_model(self, episode):
        logger.info("[CheckpointDisabled] skip final_model.pth at ep=%s", episode)
