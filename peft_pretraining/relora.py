import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig


@dataclass
class ReLoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    keep_original: bool


class ReLoRaModel(torch.nn.Module):
    def __init__(self, model, r, lora_alpha, lora_dropout, target_modules, keep_original=True):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.keep_original = keep_original
        self._config = ReLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            keep_original=keep_original,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            new_module = ReLoRaLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                keep_original=self.keep_original,
            )
            if self.keep_original:
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias

            if getattr(module, "state", None) is not None:
                new_module.state = module.state
                new_module.to(module.weight.device)

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge_and_reinit(self):
        for module in self.modules():
            if isinstance(module, ReLoRaLinear):
                module.merge_and_reinit()

    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "relora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)
    
    @classmethod
    def from_pretrained(cls, path):
        # NOT TESTED
        with open(os.path.join(path, "relora_config.json"), "r") as f:
            relora_config = json.load(f)
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        base_model = AutoModel.from_config(config)
        model = cls(base_model, **relora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        
        model.load_state_dict(state_dict, strict=True)
        return model


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class ReLoRaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        keep_original: bool = True,
        **kwargs,
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        if keep_original:
            nn.Linear.__init__(self, in_features, out_features, **kwargs)
        else:
            nn.Module.__init__(self)
            self.weight = None
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.keep_original = keep_original

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            if self.keep_original:
                self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if not hasattr(self, "lora_A"):
            # we are in nn.Linear calling reset_parameters
            nn.Linear.reset_parameters(self)
            return

        if self.keep_original:
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            nn.Linear.reset_parameters(self)
            return

        # disgard original, but now we need to init both A and B with kaiming
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

    def merge_and_reinit(self):
        if not self.keep_original:
            print("WARNING: Skipping merge and reinit, because keep_original is False")
            return

        self.weight.data += self.lora_B.weight @ self.lora_A.weight * self.scaling
        self.merged = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        if not self.keep_original:
            # just lora
            return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        result = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result
