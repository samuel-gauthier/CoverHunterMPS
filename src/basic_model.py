#!/usr/bin/env python3
# author:liufeng
# datetime:2022/8/26 9:57 AM
# software: PyCharm

import logging
from abc import abstractmethod
from typing import Dict, Tuple

import torch

from src.pytorch_utils import get_latest_model, get_model_with_epoch


class BasicModel(torch.nn.Module):
    def __init__(self, hp) -> None:
        super().__init__()
        self._hp = hp
        self._epoch = 0
        self._step = 0

    def load_model_parameters(
        self, model_dir, epoch_num=-1, device="mps", advanced=False,
    ):
        """load parameters from pt model, and return model epoch,
        if advanced, model can has different variables from saved"""
        if epoch_num == -1:
            model_path, epoch_num = get_latest_model(model_dir, "g_")
        else:
            model_path = get_model_with_epoch(model_dir, "g_", epoch_num)
            assert model_path, f"Error:model with epoch {epoch_num} not found"

        state_dict_g = torch.load(model_path, map_location=device)["generator"]
        if advanced:
            model_dict = self.state_dict()
            valid_dict = {
                k: v for k, v in state_dict_g.items() if k in model_dict
            }
            model_dict.update(valid_dict)
            self.load_state_dict(model_dict)
            for k in model_dict:
                if k not in state_dict_g:
                    logging.warning(f"{k} not be initialized")
        else:
            self.load_state_dict(state_dict_g)

        self.eval()
        self._epoch = epoch_num
        logging.info(
            f"Successful init model with epoch-{self._epoch}, device:{device}\n",
        )
        return self._epoch

    def save_model_parameters(self, g_checkpoint_path) -> None:
        torch.save({"generator": self.state_dict()}, g_checkpoint_path)

    def get_epoch_num(self):
        return self._epoch

    def get_step_num(self):
        return self._step

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self, anchor: torch.Tensor, label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        pass

    @abstractmethod
    def inference(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    # @torch.jit.export
    def get_embed_length(self) -> int:
        return self._hp["embed_dim"]

    def get_ce_embed_length(self) -> int:
        return self._hp["ce"]["output_dims"]

    def model_size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def dump_torch_script(self, dump_path) -> None:
        script_model = torch.jit.script(self)
        script_model.save(dump_path)
        logging.info(f"Export model successfully, see {dump_path}")


if __name__ == "__main__":
    pass
