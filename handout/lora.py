import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        
        #TODO: Initialize the inherited class, nn.linear 
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.has_weights_merged = False

        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
            self.lora_scaling = lora_alpha/lora_rank

            #TODO: Fill in the "..."
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            #TODO: Initialize both lora_A and lora_B with torch.nn.init. Refer to the paper to see how each is initialize
            #Hint: lora_A is initialized using kaiming_uniform_ using negative slope (a) as math.sqrt(5)

            nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #TODO: return input after the forward pass
        #Hint: Make sure you to merge in LORA matrices only if not already merged 
        # raise NotImplementedError

        if self.is_lora():
            if not self.has_weights_merged:
                lora_output = self.lora_B @ self.lora_A
                if self.lora_dropout:
                    lora_output = self.lora_dropout(lora_output)
                lora_output = input @ lora_output.T
                # print(f"input shape: {input.shape}")  # Debugging line
                # print(f"lora_B shape: {self.lora_B.shape}")  # Debugging line
                # print(f"lora_A shape: {self.lora_A.shape}")  # Debugging line
                # print(f"lora_output shape: {lora_output.shape}")  # Debugging line
                out = self.lora_scaling * lora_output
                return super().forward(input) + out
            else:
                return super().forward(input)
        else:
            return super().forward(input)

    def train(self, mode: bool = True) -> "LoRALinear":
        #TODO: Set the linear layer into train mode
        #Hint: Make sure to demerge LORA matrices if already merged
        # raise NotImplementedError
        # return self

        super().train(mode)
        if self.is_lora() and self.has_weights_merged:
            self.weight.data = self.weight.data - self.lora_scaling*(self.lora_B @ self.lora_A)
            self.has_weights_merged = False
        return self
            

    def eval(self) -> "LoRALinear":
        #TODO: Set the linear layer into eval mode
        #Hint: Make sure to merge LORA matrices if already demerged

        super().eval()
        if self.is_lora() and not self.has_weights_merged:
            self.weight.data = self.weight.data + self.lora_scaling*(self.lora_B @ self.lora_A)
            self.has_weights_merged = True
        return self
    
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    #TODO: Loop through parameters and mark some as trainable. Which ones should these be?
    #Hint: How do you mark a parameter as trainable (or not trainable)?
    # raise NotImplementedError

    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules(): 
        if isinstance(module, LoRALinear) and module.is_lora():
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    return model
