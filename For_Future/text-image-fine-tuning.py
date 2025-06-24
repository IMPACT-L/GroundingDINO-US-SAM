#%% CHANGE GroundingDINO
# CHANGE Transformer


# UNFREEZE BERT:
for param in model.bert.parameters():
    param.requires_grad = True  # Fine-tune BERT

# Optional: Use LoRA for lightweight BERT tuning
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"])
model.bert = get_peft_model(model.bert, config)

# Fine-tune Linear Projection (already unfrozen by default)
# FREEZE everything else (image backbone + transformer):
for name, param in model.named_parameters():
    if not any(n in name for n in ["bert", "feat_map"]):
        param.requires_grad = False
