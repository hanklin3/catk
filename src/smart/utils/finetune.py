import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def set_model_for_finetuning(model: torch.nn.Module, finetune: bool) -> None:
    def _unfreeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = True

    if finetune:
        for p in model.parameters():
            p.requires_grad = False

        try:
            _unfreeze(model.agent_encoder.token_predict_head)
            log.info("Unfreezing token_predict_head")
        except:
            log.info("No token_predict_head in model.agent_encoder")

        try:
            _unfreeze(model.agent_encoder.gmm_logits_head)
            _unfreeze(model.agent_encoder.gmm_pose_head)
            # _unfreeze(model.agent_encoder.gmm_gmm_covpose_head)
            log.info("Unfreezing gmm heads")
        except:
            log.info("No gmm_logits_head in model.agent_encoder")

        _unfreeze(model.agent_encoder.t_attn_layers)
        _unfreeze(model.agent_encoder.pt2a_attn_layers)
        _unfreeze(model.agent_encoder.a2a_attn_layers)
