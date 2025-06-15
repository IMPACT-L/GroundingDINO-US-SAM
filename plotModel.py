#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import numpy as np

#%% Create figure
fig, ax = plt.subplots(figsize=(20, 15))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Colors
backbone_color = '#FFDDC1'
transformer_color = '#C1FFD7'
bert_color = '#C1D8FF'
fusion_color = '#E8C1FF'
output_color = '#FFC1C1'

# Main components
components = [
    {"name": "Backbone\n(Swin Transformer)", "x": 10, "y": 70, "width": 20, "height": 20, "color": backbone_color},
    {"name": "BERT\nText Encoder", "x": 10, "y": 30, "width": 20, "height": 20, "color": bert_color},
    {"name": "Feature\nProjection", "x": 40, "y": 70, "width": 15, "height": 10, "color": '#FFEEBA'},
    {"name": "Transformer\nEncoder", "x": 40, "y": 50, "width": 15, "height": 20, "color": transformer_color},
    {"name": "Text\nLayers", "x": 65, "y": 60, "width": 15, "height": 10, "color": bert_color},
    {"name": "Fusion\nLayers", "x": 65, "y": 40, "width": 15, "height": 10, "color": fusion_color},
    {"name": "Transformer\nDecoder", "x": 40, "y": 20, "width": 15, "height": 20, "color": transformer_color},
    {"name": "BBox Predictions", "x": 70, "y": 20, "width": 15, "height": 10, "color": output_color},
    {"name": "Class Predictions", "x": 70, "y": 10, "width": 15, "height": 10, "color": output_color}
]

# Draw components
for comp in components:
    ax.add_patch(Rectangle((comp["x"], comp["y"]), comp["width"], comp["height"],
                 facecolor=comp["color"], edgecolor='black', lw=1.5))
    ax.text(comp["x"] + comp["width"]/2, comp["y"] + comp["height"]/2, comp["name"],
            ha='center', va='center', fontsize=9 if 'Layers' in comp["name"] else 10)

# Arrows/connections
arrows = [
    {"start": (20, 80), "end": (40, 75), "text": "Image Features"},
    {"start": (20, 40), "end": (40, 45), "text": "Text Features"},
    {"start": (55, 70), "end": (55, 65), "text": ""},
    {"start": (55, 50), "end": (55, 45), "text": ""},
    {"start": (65, 65), "end": (65, 50), "text": "Bi-attention"},
    {"start": (55, 30), "end": (55, 25), "text": ""},
    {"start": (55, 20), "end": (70, 25), "text": "Box Reg"},
    {"start": (55, 15), "end": (70, 15), "text": "Class Pred"}
]

for arrow in arrows:
    ax.annotate("", xy=arrow["end"], xytext=arrow["start"],
                arrowprops=dict(arrowstyle="->", lw=1.5))
    if arrow["text"]:
        ax.text((arrow["start"][0]+arrow["end"][0])/2, 
                (arrow["start"][1]+arrow["end"][1])/2,
                arrow["text"], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, pad=2))

# Detailed encoder blocks
for i in range(6):
    ax.add_patch(Rectangle((42, 52+i*3), 10, 2, facecolor='#A0E8B0', edgecolor='black'))
    ax.text(47, 53+i*3, f"Enc Layer {i+1}", ha='center', va='center', fontsize=8)

# Detailed decoder blocks
for i in range(6):
    ax.add_patch(Rectangle((42, 22+i*3), 10, 2, facecolor='#A0E8B0', edgecolor='black'))
    ax.text(47, 23+i*3, f"Dec Layer {i+1}", ha='center', va='center', fontsize=8)

# Title and legend
ax.text(50, 95, "GroundingDINO Architecture Overview", ha='center', va='center', fontsize=14, weight='bold')
ax.text(80, 85, "Key Components:", fontsize=10, weight='bold')
legend_elements = [
    Rectangle((0,0), 1, 1, fc=backbone_color, ec='black', label='Vision Backbone'),
    Rectangle((0,0), 1, 1, fc=bert_color, ec='black', label='Text Encoder'),
    Rectangle((0,0), 1, 1, fc=transformer_color, ec='black', label='Transformer'),
    Rectangle((0,0), 1, 1, fc=fusion_color, ec='black', label='Fusion Layers'),
    Rectangle((0,0), 1, 1, fc=output_color, ec='black', label='Output Heads')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("grounding_dino_architecture.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
🔥 model.transformer.encoder.layers.0.linear1.lora_A.default.weight
🔥 model.transformer.encoder.layers.0.linear1.lora_B.default.weight
🔥 model.transformer.encoder.layers.0.linear2.lora_A.default.weight
🔥 model.transformer.encoder.layers.0.linear2.lora_B.default.weight

🔥 model.transformer.encoder.layers.1.linear1.lora_A.default.weight
🔥 model.transformer.encoder.layers.1.linear1.lora_B.default.weight
🔥 model.transformer.encoder.layers.1.linear2.lora_A.default.weight
🔥 model.transformer.encoder.layers.1.linear2.lora_B.default.weight
🔥 model.transformer.encoder.layers.2.linear1.lora_A.default.weight
🔥 model.transformer.encoder.layers.2.linear1.lora_B.default.weight
🔥 model.transformer.encoder.layers.2.linear2.lora_A.default.weight
🔥 model.transformer.encoder.layers.2.linear2.lora_B.default.weight

🔥 model.transformer.encoder.layers.3.linear1.lora_A.default.weight
🔥 model.transformer.encoder.layers.3.linear1.lora_B.default.weight
🔥 model.transformer.encoder.layers.3.linear2.lora_A.default.weight
🔥 model.transformer.encoder.layers.3.linear2.lora_B.default.weight

🔥 model.transformer.encoder.layers.4.linear1.lora_A.default.weight
🔥 model.transformer.encoder.layers.4.linear1.lora_B.default.weight
🔥 model.transformer.encoder.layers.4.linear2.lora_A.default.weight
🔥 model.transformer.encoder.layers.4.linear2.lora_B.default.weight
🔥 model.transformer.encoder.layers.5.linear1.lora_A.default.weight
🔥 model.transformer.encoder.layers.5.linear1.lora_B.default.weight
🔥 model.transformer.encoder.layers.5.linear2.lora_A.default.weight
🔥 model.transformer.encoder.layers.5.linear2.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.0.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.0.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.0.linear1.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.0.linear1.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.0.linear2.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.0.linear2.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.1.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.1.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.1.linear1.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.1.linear1.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.1.linear2.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.1.linear2.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.2.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.2.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.2.linear1.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.2.linear1.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.2.linear2.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.2.linear2.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.3.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.3.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.3.linear1.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.3.linear1.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.3.linear2.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.3.linear2.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.4.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.4.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.4.linear1.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.4.linear1.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.4.linear2.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.4.linear2.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.5.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.5.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.5.linear1.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.5.linear1.lora_B.default.weight
🔥 model.transformer.encoder.text_layers.5.linear2.lora_A.default.weight
🔥 model.transformer.encoder.text_layers.5.linear2.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.cross_attn.sampling_offsets.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.cross_attn.attention_weights.lora_A.default.weight
🔥 model.transformer.decoder.layers.0.cross_attn.attention_weights.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.cross_attn.value_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.0.cross_attn.value_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.cross_attn.output_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.0.cross_attn.output_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.ca_text.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.0.ca_text.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.0.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.linear1.lora_A.default.weight
🔥 model.transformer.decoder.layers.0.linear1.lora_B.default.weight
🔥 model.transformer.decoder.layers.0.linear2.lora_A.default.weight
🔥 model.transformer.decoder.layers.0.linear2.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.sampling_offsets.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.sampling_offsets.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.attention_weights.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.attention_weights.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.value_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.value_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.output_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.cross_attn.output_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.ca_text.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.ca_text.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.linear1.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.linear1.lora_B.default.weight
🔥 model.transformer.decoder.layers.1.linear2.lora_A.default.weight
🔥 model.transformer.decoder.layers.1.linear2.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.sampling_offsets.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.sampling_offsets.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.attention_weights.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.attention_weights.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.value_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.value_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.output_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.cross_attn.output_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.ca_text.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.ca_text.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.linear1.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.linear1.lora_B.default.weight
🔥 model.transformer.decoder.layers.2.linear2.lora_A.default.weight
🔥 model.transformer.decoder.layers.2.linear2.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.sampling_offsets.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.sampling_offsets.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.attention_weights.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.attention_weights.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.value_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.value_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.output_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.cross_attn.output_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.ca_text.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.ca_text.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.linear1.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.linear1.lora_B.default.weight
🔥 model.transformer.decoder.layers.3.linear2.lora_A.default.weight
🔥 model.transformer.decoder.layers.3.linear2.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.sampling_offsets.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.sampling_offsets.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.attention_weights.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.attention_weights.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.value_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.value_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.output_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.cross_attn.output_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.ca_text.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.ca_text.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.linear1.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.linear1.lora_B.default.weight
🔥 model.transformer.decoder.layers.4.linear2.lora_A.default.weight
🔥 model.transformer.decoder.layers.4.linear2.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.sampling_offsets.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.sampling_offsets.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.attention_weights.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.attention_weights.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.value_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.value_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.output_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.cross_attn.output_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.ca_text.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.ca_text.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.self_attn.out_proj.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.self_attn.out_proj.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.linear1.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.linear1.lora_B.default.weight
🔥 model.transformer.decoder.layers.5.linear2.lora_A.default.weight
🔥 model.transformer.decoder.layers.5.linear2.lora_B.default.weight
🔥 model.transformer.decoder.bbox_embed.0.layers.0.lora_A.default.weight
🔥 model.transformer.decoder.bbox_embed.0.layers.0.lora_B.default.weight
🔥 model.transformer.decoder.bbox_embed.0.layers.1.lora_A.default.weight
🔥 model.transformer.decoder.bbox_embed.0.layers.1.lora_B.default.weight
🔥 model.transformer.decoder.bbox_embed.0.layers.2.modules_to_save.default.weight
🔥 model.transformer.decoder.bbox_embed.0.layers.2.modules_to_save.default.bias
🔥 model.feat_map.lora_A.default.weight
🔥 model.feat_map.lora_B.default.weight