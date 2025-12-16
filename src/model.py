"""GPT-2 model wrappers with Hierarchical Softmax head."""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GenerationConfig

from .hsm import HierarchicalSoftmaxHead


class GPT2WithHSM(nn.Module):
    """GPT-2 model with Hierarchical Softmax output head."""

    def __init__(
        self,
        root,
        tokenizer,
        hidden_size: int = 768,
        pretrained: bool = True,
        freeze_transformer: bool = False,
        use_prf: bool = False,
        num_random_features: int = 256,
        model_name: str = "gpt2",
    ):
        super().__init__()

        self.pretrained = pretrained
        self.freeze_transformer = freeze_transformer
        self.use_prf = use_prf

        if pretrained:
            self.transformer = GPT2Model.from_pretrained(model_name)
            print(f"Loaded pretrained weights from '{model_name}'")
        else:
            config = GPT2Config.from_pretrained(model_name)
            self.transformer = GPT2Model(config)
            print(f"Initialized random GPT-2 weights")

        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
            print("Transformer weights frozen")

        self.hsm_head = HierarchicalSoftmaxHead(
            root=root,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            use_prf=use_prf,
            num_random_features=num_random_features,
        )

        self.generation_config = GenerationConfig()
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: (b, s)
            attention_mask: (b, s)
            labels: (b, s) target token IDs

        Returns:
            Dict with "loss" (if labels) and "hidden_states"
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        result = {"hidden_states": hidden_states}

        if labels is not None:
            shifted_hidden = hidden_states[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = self.hsm_head(shifted_hidden, shifted_labels)
            result["loss"] = loss

        return result

    def get_config_dict(self) -> dict:
        return {
            "pretrained": self.pretrained,
            "freeze_transformer": self.freeze_transformer,
            "use_prf": self.use_prf,
            "num_random_features": self.hsm_head.num_random_features if self.use_prf else None,
            "hidden_size": self.hsm_head.hidden_size,
            "vocab_size": self.hsm_head.vocab_size,
            "num_internal_nodes": self.hsm_head.num_internal_nodes,
        }


class GPT2WithSoftmax(nn.Module):
    """Standard GPT-2 model with softmax output (baseline)."""

    def __init__(
        self,
        vocab_size: int,
        pretrained: bool = True,
        freeze_transformer: bool = False,
        model_name: str = "gpt2",
    ):
        super().__init__()

        self.pretrained = pretrained
        self.freeze_transformer = freeze_transformer
        self.vocab_size = vocab_size

        if pretrained:
            self.transformer = GPT2Model.from_pretrained(model_name)
            config = self.transformer.config
        else:
            config = GPT2Config.from_pretrained(model_name)
            self.transformer = GPT2Model(config)

        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.generation_config = GenerationConfig()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        result = {"hidden_states": hidden_states}

        if labels is not None:
            shifted_hidden = hidden_states[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

            logits = self.lm_head(shifted_hidden)
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = shifted_labels.view(-1)

            loss = nn.functional.cross_entropy(logits_flat, labels_flat)
            result["loss"] = loss
            result["logits"] = logits

        return result


def create_model(root, tokenizer, config: dict) -> nn.Module:
    """Factory function to create model from config."""
    return GPT2WithHSM(
        root=root,
        tokenizer=tokenizer,
        hidden_size=768,
        pretrained=config.get("pretrained", True),
        freeze_transformer=config.get("freeze_transformer", False),
        use_prf=config.get("use_prf", False),
        num_random_features=config.get("num_random_features", 256),
    )
