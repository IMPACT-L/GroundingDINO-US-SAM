SAM2Base(
  (image_encoder): ImageEncoder(
    (trunk): Hiera(
      (patch_embed): PatchEmbed(
        (proj): Conv2d(3, 144, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
      )
      (blocks): ModuleList(
        (0-1): 2 x MultiScaleBlock(
          (norm1): LayerNorm((144,), eps=1e-06, elementwise_affine=True)
          (attn): MultiScaleAttention(
            (qkv): Linear(in_features=144, out_features=432, bias=True)
            (proj): Linear(in_features=144, out_features=144, bias=True)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((144,), eps=1e-06, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=144, out_features=576, bias=True)
              (1): Linear(in_features=576, out_features=144, bias=True)
            )
            (act): GELU(approximate='none')
          )
        )
        (2): MultiScaleBlock(
          (norm1): LayerNorm((144,), eps=1e-06, elementwise_affine=True)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (attn): MultiScaleAttention(
            (q_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            (qkv): Linear(in_features=144, out_features=864, bias=True)
            (proj): Linear(in_features=288, out_features=288, bias=True)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((288,), eps=1e-06, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=288, out_features=1152, bias=True)
              (1): Linear(in_features=1152, out_features=288, bias=True)
            )
            (act): GELU(approximate='none')
          )
          (proj): Linear(in_features=144, out_features=288, bias=True)
        )
        (3-7): 5 x MultiScaleBlock(
          (norm1): LayerNorm((288,), eps=1e-06, elementwise_affine=True)
          (attn): MultiScaleAttention(
            (qkv): Linear(in_features=288, out_features=864, bias=True)
            (proj): Linear(in_features=288, out_features=288, bias=True)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((288,), eps=1e-06, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=288, out_features=1152, bias=True)
              (1): Linear(in_features=1152, out_features=288, bias=True)
            )
            (act): GELU(approximate='none')
          )
        )
        (8): MultiScaleBlock(
          (norm1): LayerNorm((288,), eps=1e-06, elementwise_affine=True)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (attn): MultiScaleAttention(
            (q_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            (qkv): Linear(in_features=288, out_features=1728, bias=True)
            (proj): Linear(in_features=576, out_features=576, bias=True)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((576,), eps=1e-06, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=576, out_features=2304, bias=True)
              (1): Linear(in_features=2304, out_features=576, bias=True)
            )
            (act): GELU(approximate='none')
          )
          (proj): Linear(in_features=288, out_features=576, bias=True)
        )
        (9-43): 35 x MultiScaleBlock(
          (norm1): LayerNorm((576,), eps=1e-06, elementwise_affine=True)
          (attn): MultiScaleAttention(
            (qkv): Linear(in_features=576, out_features=1728, bias=True)
            (proj): Linear(in_features=576, out_features=576, bias=True)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((576,), eps=1e-06, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=576, out_features=2304, bias=True)
              (1): Linear(in_features=2304, out_features=576, bias=True)
            )
            (act): GELU(approximate='none')
          )
        )
        (44): MultiScaleBlock(
          (norm1): LayerNorm((576,), eps=1e-06, elementwise_affine=True)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (attn): MultiScaleAttention(
            (q_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            (qkv): Linear(in_features=576, out_features=3456, bias=True)
            (proj): Linear(in_features=1152, out_features=1152, bias=True)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=1152, out_features=4608, bias=True)
              (1): Linear(in_features=4608, out_features=1152, bias=True)
            )
            (act): GELU(approximate='none')
          )
          (proj): Linear(in_features=576, out_features=1152, bias=True)
        )
        (45-47): 3 x MultiScaleBlock(
          (norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (attn): MultiScaleAttention(
            (qkv): Linear(in_features=1152, out_features=3456, bias=True)
            (proj): Linear(in_features=1152, out_features=1152, bias=True)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=1152, out_features=4608, bias=True)
              (1): Linear(in_features=4608, out_features=1152, bias=True)
            )
            (act): GELU(approximate='none')
          )
        )
      )
    )
    (neck): FpnNeck(
      (position_encoding): PositionEmbeddingSine()
      (convs): ModuleList(
        (0): Sequential(
          (conv): Conv2d(1152, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Sequential(
          (conv): Conv2d(576, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Sequential(
          (conv): Conv2d(288, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (3): Sequential(
          (conv): Conv2d(144, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mask_downsample): Conv2d(1, 1, kernel_size=(4, 4), stride=(4, 4))
  (memory_attention): MemoryAttention(
    (layers): ModuleList(
      (0-3): 4 x MemoryAttentionLayer(
        (self_attn): RoPEAttention(
          (q_proj): Linear(in_features=256, out_features=256, bias=True)
          (k_proj): Linear(in_features=256, out_features=256, bias=True)
          (v_proj): Linear(in_features=256, out_features=256, bias=True)
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (cross_attn_image): RoPEAttention(
          (q_proj): Linear(in_features=256, out_features=256, bias=True)
          (k_proj): Linear(in_features=64, out_features=256, bias=True)
          (v_proj): Linear(in_features=64, out_features=256, bias=True)
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
    )
    (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  )
  (memory_encoder): MemoryEncoder(
    (mask_downsampler): MaskDownSampler(
      (encoder): Sequential(
        (0): Conv2d(1, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): LayerNorm2d()
        (2): GELU(approximate='none')
        (3): Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (4): LayerNorm2d()
        (5): GELU(approximate='none')
        (6): Conv2d(16, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (7): LayerNorm2d()
        (8): GELU(approximate='none')
        (9): Conv2d(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (10): LayerNorm2d()
        (11): GELU(approximate='none')
        (12): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (pix_feat_proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (fuser): Fuser(
      (proj): Identity()
      (layers): ModuleList(
        (0-1): 2 x CXBlock(
          (dwconv): Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=256)
          (norm): LayerNorm2d()
          (pwconv1): Linear(in_features=256, out_features=1024, bias=True)
          (act): GELU(approximate='none')
          (pwconv2): Linear(in_features=1024, out_features=256, bias=True)
          (drop_path): Identity()
        )
      )
    )
    (position_encoding): PositionEmbeddingSine()
    (out_proj): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
  )
  (sam_prompt_encoder): PromptEncoder(
    (pe_layer): PositionEmbeddingRandom()
    (point_embeddings): ModuleList(
      (0-3): 4 x Embedding(1, 256)
    )
    (not_a_point_embed): Embedding(1, 256)
    (mask_downscaling): Sequential(
      (0): Conv2d(1, 4, kernel_size=(2, 2), stride=(2, 2))
      (1): LayerNorm2d()
      (2): GELU(approximate='none')
      (3): Conv2d(4, 16, kernel_size=(2, 2), stride=(2, 2))
      (4): LayerNorm2d()
      (5): GELU(approximate='none')
      (6): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
    )
    (no_mask_embed): Embedding(1, 256)
  )
  (sam_mask_decoder): MaskDecoder(
    (transformer): TwoWayTransformer(
      (layers): ModuleList(
        (0-1): 2 x TwoWayAttentionBlock(
          (self_attn): Attention(
            (q_proj): Linear(in_features=256, out_features=256, bias=True)
            (k_proj): Linear(in_features=256, out_features=256, bias=True)
            (v_proj): Linear(in_features=256, out_features=256, bias=True)
            (out_proj): Linear(in_features=256, out_features=256, bias=True)
          )
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (cross_attn_token_to_image): Attention(
            (q_proj): Linear(in_features=256, out_features=128, bias=True)
            (k_proj): Linear(in_features=256, out_features=128, bias=True)
            (v_proj): Linear(in_features=256, out_features=128, bias=True)
            (out_proj): Linear(in_features=128, out_features=256, bias=True)
          )
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (mlp): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): Linear(in_features=2048, out_features=256, bias=True)
            )
            (act): ReLU()
          )
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (cross_attn_image_to_token): Attention(
            (q_proj): Linear(in_features=256, out_features=128, bias=True)
            (k_proj): Linear(in_features=256, out_features=128, bias=True)
            (v_proj): Linear(in_features=256, out_features=128, bias=True)
            (out_proj): Linear(in_features=128, out_features=256, bias=True)
          )
        )
      )
      (final_attn_token_to_image): Attention(
        (q_proj): Linear(in_features=256, out_features=128, bias=True)
        (k_proj): Linear(in_features=256, out_features=128, bias=True)
        (v_proj): Linear(in_features=256, out_features=128, bias=True)
        (out_proj): Linear(in_features=128, out_features=256, bias=True)
      )
      (norm_final_attn): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (iou_token): Embedding(1, 256)
    (mask_tokens): Embedding(4, 256)
    (obj_score_token): Embedding(1, 256)
    (output_upscaling): Sequential(
      (0): ConvTranspose2d(256, 64, kernel_size=(2, 2), stride=(2, 2))
      (1): LayerNorm2d()
      (2): GELU(approximate='none')
      (3): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
      (4): GELU(approximate='none')
    )
    (conv_s0): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
    (conv_s1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    (output_hypernetworks_mlps): ModuleList(
      (0-3): 4 x MLP(
        (layers): ModuleList(
          (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
          (2): Linear(in_features=256, out_features=32, bias=True)
        )
        (act): ReLU()
      )
    )
    (iou_prediction_head): MLP(
      (layers): ModuleList(
        (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
        (2): Linear(in_features=256, out_features=4, bias=True)
      )
      (act): ReLU()
    )
    (pred_obj_score_head): MLP(
      (layers): ModuleList(
        (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
        (2): Linear(in_features=256, out_features=1, bias=True)
      )
      (act): ReLU()
    )
  )
  (obj_ptr_proj): MLP(
    (layers): ModuleList(
      (0-2): 3 x Linear(in_features=256, out_features=256, bias=True)
    )
    (act): ReLU()
  )
  (obj_ptr_tpos_proj): Linear(in_features=256, out_features=64, bias=True)
)


# === Component Status ===

# base_model:
  ❄️ model.transformer.level_embed
  ❄️ model.transformer.encoder.layers.0.self_attn.sampling_offsets.weight
  ❄️ model.transformer.encoder.layers.0.self_attn.sampling_offsets.bias
  ❄️ model.transformer.encoder.layers.0.self_attn.attention_weights.weight
  ❄️ model.transformer.encoder.layers.0.self_attn.attention_weights.bias
  ❄️ model.transformer.encoder.layers.0.self_attn.value_proj.weight
  ❄️ model.transformer.encoder.layers.0.self_attn.value_proj.bias
  ❄️ model.transformer.encoder.layers.0.self_attn.output_proj.weight
  ❄️ model.transformer.encoder.layers.0.self_attn.output_proj.bias
  ❄️ model.transformer.encoder.layers.0.norm1.weight
  ❄️ model.transformer.encoder.layers.0.norm1.bias
  ❄️ model.transformer.encoder.layers.0.linear1.base_layer.weight
  ❄️ model.transformer.encoder.layers.0.linear1.base_layer.bias
  🔥 model.transformer.encoder.layers.0.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.layers.0.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.0.linear2.base_layer.weight
  ❄️ model.transformer.encoder.layers.0.linear2.base_layer.bias
  🔥 model.transformer.encoder.layers.0.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.layers.0.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.0.norm2.weight
  ❄️ model.transformer.encoder.layers.0.norm2.bias
  ❄️ model.transformer.encoder.layers.1.self_attn.sampling_offsets.weight
  ❄️ model.transformer.encoder.layers.1.self_attn.sampling_offsets.bias
  ❄️ model.transformer.encoder.layers.1.self_attn.attention_weights.weight
  ❄️ model.transformer.encoder.layers.1.self_attn.attention_weights.bias
  ❄️ model.transformer.encoder.layers.1.self_attn.value_proj.weight
  ❄️ model.transformer.encoder.layers.1.self_attn.value_proj.bias
  ❄️ model.transformer.encoder.layers.1.self_attn.output_proj.weight
  ❄️ model.transformer.encoder.layers.1.self_attn.output_proj.bias
  ❄️ model.transformer.encoder.layers.1.norm1.weight
  ❄️ model.transformer.encoder.layers.1.norm1.bias
  ❄️ model.transformer.encoder.layers.1.linear1.base_layer.weight
  ❄️ model.transformer.encoder.layers.1.linear1.base_layer.bias
  🔥 model.transformer.encoder.layers.1.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.layers.1.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.1.linear2.base_layer.weight
  ❄️ model.transformer.encoder.layers.1.linear2.base_layer.bias
  🔥 model.transformer.encoder.layers.1.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.layers.1.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.1.norm2.weight
  ❄️ model.transformer.encoder.layers.1.norm2.bias
  ❄️ model.transformer.encoder.layers.2.self_attn.sampling_offsets.weight
  ❄️ model.transformer.encoder.layers.2.self_attn.sampling_offsets.bias
  ❄️ model.transformer.encoder.layers.2.self_attn.attention_weights.weight
  ❄️ model.transformer.encoder.layers.2.self_attn.attention_weights.bias
  ❄️ model.transformer.encoder.layers.2.self_attn.value_proj.weight
  ❄️ model.transformer.encoder.layers.2.self_attn.value_proj.bias
  ❄️ model.transformer.encoder.layers.2.self_attn.output_proj.weight
  ❄️ model.transformer.encoder.layers.2.self_attn.output_proj.bias
  ❄️ model.transformer.encoder.layers.2.norm1.weight
  ❄️ model.transformer.encoder.layers.2.norm1.bias
  ❄️ model.transformer.encoder.layers.2.linear1.base_layer.weight
  ❄️ model.transformer.encoder.layers.2.linear1.base_layer.bias
  🔥 model.transformer.encoder.layers.2.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.layers.2.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.2.linear2.base_layer.weight
  ❄️ model.transformer.encoder.layers.2.linear2.base_layer.bias
  🔥 model.transformer.encoder.layers.2.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.layers.2.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.2.norm2.weight
  ❄️ model.transformer.encoder.layers.2.norm2.bias
  ❄️ model.transformer.encoder.layers.3.self_attn.sampling_offsets.weight
  ❄️ model.transformer.encoder.layers.3.self_attn.sampling_offsets.bias
  ❄️ model.transformer.encoder.layers.3.self_attn.attention_weights.weight
  ❄️ model.transformer.encoder.layers.3.self_attn.attention_weights.bias
  ❄️ model.transformer.encoder.layers.3.self_attn.value_proj.weight
  ❄️ model.transformer.encoder.layers.3.self_attn.value_proj.bias
  ❄️ model.transformer.encoder.layers.3.self_attn.output_proj.weight
  ❄️ model.transformer.encoder.layers.3.self_attn.output_proj.bias
  ❄️ model.transformer.encoder.layers.3.norm1.weight
  ❄️ model.transformer.encoder.layers.3.norm1.bias
  ❄️ model.transformer.encoder.layers.3.linear1.base_layer.weight
  ❄️ model.transformer.encoder.layers.3.linear1.base_layer.bias
  🔥 model.transformer.encoder.layers.3.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.layers.3.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.3.linear2.base_layer.weight
  ❄️ model.transformer.encoder.layers.3.linear2.base_layer.bias
  🔥 model.transformer.encoder.layers.3.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.layers.3.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.3.norm2.weight
  ❄️ model.transformer.encoder.layers.3.norm2.bias
  ❄️ model.transformer.encoder.layers.4.self_attn.sampling_offsets.weight
  ❄️ model.transformer.encoder.layers.4.self_attn.sampling_offsets.bias
  ❄️ model.transformer.encoder.layers.4.self_attn.attention_weights.weight
  ❄️ model.transformer.encoder.layers.4.self_attn.attention_weights.bias
  ❄️ model.transformer.encoder.layers.4.self_attn.value_proj.weight
  ❄️ model.transformer.encoder.layers.4.self_attn.value_proj.bias
  ❄️ model.transformer.encoder.layers.4.self_attn.output_proj.weight
  ❄️ model.transformer.encoder.layers.4.self_attn.output_proj.bias
  ❄️ model.transformer.encoder.layers.4.norm1.weight
  ❄️ model.transformer.encoder.layers.4.norm1.bias
  ❄️ model.transformer.encoder.layers.4.linear1.base_layer.weight
  ❄️ model.transformer.encoder.layers.4.linear1.base_layer.bias
  🔥 model.transformer.encoder.layers.4.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.layers.4.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.4.linear2.base_layer.weight
  ❄️ model.transformer.encoder.layers.4.linear2.base_layer.bias
  🔥 model.transformer.encoder.layers.4.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.layers.4.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.4.norm2.weight
  ❄️ model.transformer.encoder.layers.4.norm2.bias
  ❄️ model.transformer.encoder.layers.5.self_attn.sampling_offsets.weight
  ❄️ model.transformer.encoder.layers.5.self_attn.sampling_offsets.bias
  ❄️ model.transformer.encoder.layers.5.self_attn.attention_weights.weight
  ❄️ model.transformer.encoder.layers.5.self_attn.attention_weights.bias
  ❄️ model.transformer.encoder.layers.5.self_attn.value_proj.weight
  ❄️ model.transformer.encoder.layers.5.self_attn.value_proj.bias
  ❄️ model.transformer.encoder.layers.5.self_attn.output_proj.weight
  ❄️ model.transformer.encoder.layers.5.self_attn.output_proj.bias
  ❄️ model.transformer.encoder.layers.5.norm1.weight
  ❄️ model.transformer.encoder.layers.5.norm1.bias
  ❄️ model.transformer.encoder.layers.5.linear1.base_layer.weight
  ❄️ model.transformer.encoder.layers.5.linear1.base_layer.bias
  🔥 model.transformer.encoder.layers.5.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.layers.5.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.5.linear2.base_layer.weight
  ❄️ model.transformer.encoder.layers.5.linear2.base_layer.bias
  🔥 model.transformer.encoder.layers.5.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.layers.5.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.layers.5.norm2.weight
  ❄️ model.transformer.encoder.layers.5.norm2.bias
  ❄️ model.transformer.encoder.text_layers.0.self_attn.in_proj_weight
  ❄️ model.transformer.encoder.text_layers.0.self_attn.in_proj_bias
  ❄️ model.transformer.encoder.text_layers.0.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.0.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.encoder.text_layers.0.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.0.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.0.linear1.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.0.linear1.base_layer.bias
  🔥 model.transformer.encoder.text_layers.0.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.0.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.0.linear2.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.0.linear2.base_layer.bias
  🔥 model.transformer.encoder.text_layers.0.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.0.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.0.norm1.weight
  ❄️ model.transformer.encoder.text_layers.0.norm1.bias
  ❄️ model.transformer.encoder.text_layers.0.norm2.weight
  ❄️ model.transformer.encoder.text_layers.0.norm2.bias
  ❄️ model.transformer.encoder.text_layers.1.self_attn.in_proj_weight
  ❄️ model.transformer.encoder.text_layers.1.self_attn.in_proj_bias
  ❄️ model.transformer.encoder.text_layers.1.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.1.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.encoder.text_layers.1.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.1.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.1.linear1.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.1.linear1.base_layer.bias
  🔥 model.transformer.encoder.text_layers.1.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.1.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.1.linear2.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.1.linear2.base_layer.bias
  🔥 model.transformer.encoder.text_layers.1.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.1.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.1.norm1.weight
  ❄️ model.transformer.encoder.text_layers.1.norm1.bias
  ❄️ model.transformer.encoder.text_layers.1.norm2.weight
  ❄️ model.transformer.encoder.text_layers.1.norm2.bias
  ❄️ model.transformer.encoder.text_layers.2.self_attn.in_proj_weight
  ❄️ model.transformer.encoder.text_layers.2.self_attn.in_proj_bias
  ❄️ model.transformer.encoder.text_layers.2.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.2.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.encoder.text_layers.2.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.2.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.2.linear1.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.2.linear1.base_layer.bias
  🔥 model.transformer.encoder.text_layers.2.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.2.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.2.linear2.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.2.linear2.base_layer.bias
  🔥 model.transformer.encoder.text_layers.2.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.2.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.2.norm1.weight
  ❄️ model.transformer.encoder.text_layers.2.norm1.bias
  ❄️ model.transformer.encoder.text_layers.2.norm2.weight
  ❄️ model.transformer.encoder.text_layers.2.norm2.bias
  ❄️ model.transformer.encoder.text_layers.3.self_attn.in_proj_weight
  ❄️ model.transformer.encoder.text_layers.3.self_attn.in_proj_bias
  ❄️ model.transformer.encoder.text_layers.3.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.3.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.encoder.text_layers.3.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.3.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.3.linear1.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.3.linear1.base_layer.bias
  🔥 model.transformer.encoder.text_layers.3.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.3.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.3.linear2.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.3.linear2.base_layer.bias
  🔥 model.transformer.encoder.text_layers.3.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.3.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.3.norm1.weight
  ❄️ model.transformer.encoder.text_layers.3.norm1.bias
  ❄️ model.transformer.encoder.text_layers.3.norm2.weight
  ❄️ model.transformer.encoder.text_layers.3.norm2.bias
  ❄️ model.transformer.encoder.text_layers.4.self_attn.in_proj_weight
  ❄️ model.transformer.encoder.text_layers.4.self_attn.in_proj_bias
  ❄️ model.transformer.encoder.text_layers.4.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.4.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.encoder.text_layers.4.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.4.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.4.linear1.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.4.linear1.base_layer.bias
  🔥 model.transformer.encoder.text_layers.4.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.4.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.4.linear2.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.4.linear2.base_layer.bias
  🔥 model.transformer.encoder.text_layers.4.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.4.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.4.norm1.weight
  ❄️ model.transformer.encoder.text_layers.4.norm1.bias
  ❄️ model.transformer.encoder.text_layers.4.norm2.weight
  ❄️ model.transformer.encoder.text_layers.4.norm2.bias
  ❄️ model.transformer.encoder.text_layers.5.self_attn.in_proj_weight
  ❄️ model.transformer.encoder.text_layers.5.self_attn.in_proj_bias
  ❄️ model.transformer.encoder.text_layers.5.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.5.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.encoder.text_layers.5.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.5.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.5.linear1.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.5.linear1.base_layer.bias
  🔥 model.transformer.encoder.text_layers.5.linear1.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.5.linear1.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.5.linear2.base_layer.weight
  ❄️ model.transformer.encoder.text_layers.5.linear2.base_layer.bias
  🔥 model.transformer.encoder.text_layers.5.linear2.lora_A.default.weight
  🔥 model.transformer.encoder.text_layers.5.linear2.lora_B.default.weight
  ❄️ model.transformer.encoder.text_layers.5.norm1.weight
  ❄️ model.transformer.encoder.text_layers.5.norm1.bias
  ❄️ model.transformer.encoder.text_layers.5.norm2.weight
  ❄️ model.transformer.encoder.text_layers.5.norm2.bias
  ❄️ model.transformer.encoder.fusion_layers.0.gamma_v
  ❄️ model.transformer.encoder.fusion_layers.0.gamma_l
  ❄️ model.transformer.encoder.fusion_layers.0.layer_norm_v.weight
  ❄️ model.transformer.encoder.fusion_layers.0.layer_norm_v.bias
  ❄️ model.transformer.encoder.fusion_layers.0.layer_norm_l.weight
  ❄️ model.transformer.encoder.fusion_layers.0.layer_norm_l.bias
  ❄️ model.transformer.encoder.fusion_layers.0.attn.v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.0.attn.v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.0.attn.l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.0.attn.l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.0.attn.values_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.0.attn.values_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.0.attn.values_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.0.attn.values_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.0.attn.out_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.0.attn.out_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.0.attn.out_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.0.attn.out_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.1.gamma_v
  ❄️ model.transformer.encoder.fusion_layers.1.gamma_l
  ❄️ model.transformer.encoder.fusion_layers.1.layer_norm_v.weight
  ❄️ model.transformer.encoder.fusion_layers.1.layer_norm_v.bias
  ❄️ model.transformer.encoder.fusion_layers.1.layer_norm_l.weight
  ❄️ model.transformer.encoder.fusion_layers.1.layer_norm_l.bias
  ❄️ model.transformer.encoder.fusion_layers.1.attn.v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.1.attn.v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.1.attn.l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.1.attn.l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.1.attn.values_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.1.attn.values_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.1.attn.values_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.1.attn.values_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.1.attn.out_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.1.attn.out_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.1.attn.out_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.1.attn.out_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.2.gamma_v
  ❄️ model.transformer.encoder.fusion_layers.2.gamma_l
  ❄️ model.transformer.encoder.fusion_layers.2.layer_norm_v.weight
  ❄️ model.transformer.encoder.fusion_layers.2.layer_norm_v.bias
  ❄️ model.transformer.encoder.fusion_layers.2.layer_norm_l.weight
  ❄️ model.transformer.encoder.fusion_layers.2.layer_norm_l.bias
  ❄️ model.transformer.encoder.fusion_layers.2.attn.v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.2.attn.v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.2.attn.l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.2.attn.l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.2.attn.values_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.2.attn.values_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.2.attn.values_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.2.attn.values_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.2.attn.out_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.2.attn.out_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.2.attn.out_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.2.attn.out_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.3.gamma_v
  ❄️ model.transformer.encoder.fusion_layers.3.gamma_l
  ❄️ model.transformer.encoder.fusion_layers.3.layer_norm_v.weight
  ❄️ model.transformer.encoder.fusion_layers.3.layer_norm_v.bias
  ❄️ model.transformer.encoder.fusion_layers.3.layer_norm_l.weight
  ❄️ model.transformer.encoder.fusion_layers.3.layer_norm_l.bias
  ❄️ model.transformer.encoder.fusion_layers.3.attn.v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.3.attn.v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.3.attn.l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.3.attn.l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.3.attn.values_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.3.attn.values_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.3.attn.values_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.3.attn.values_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.3.attn.out_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.3.attn.out_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.3.attn.out_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.3.attn.out_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.4.gamma_v
  ❄️ model.transformer.encoder.fusion_layers.4.gamma_l
  ❄️ model.transformer.encoder.fusion_layers.4.layer_norm_v.weight
  ❄️ model.transformer.encoder.fusion_layers.4.layer_norm_v.bias
  ❄️ model.transformer.encoder.fusion_layers.4.layer_norm_l.weight
  ❄️ model.transformer.encoder.fusion_layers.4.layer_norm_l.bias
  ❄️ model.transformer.encoder.fusion_layers.4.attn.v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.4.attn.v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.4.attn.l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.4.attn.l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.4.attn.values_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.4.attn.values_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.4.attn.values_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.4.attn.values_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.4.attn.out_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.4.attn.out_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.4.attn.out_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.4.attn.out_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.5.gamma_v
  ❄️ model.transformer.encoder.fusion_layers.5.gamma_l
  ❄️ model.transformer.encoder.fusion_layers.5.layer_norm_v.weight
  ❄️ model.transformer.encoder.fusion_layers.5.layer_norm_v.bias
  ❄️ model.transformer.encoder.fusion_layers.5.layer_norm_l.weight
  ❄️ model.transformer.encoder.fusion_layers.5.layer_norm_l.bias
  ❄️ model.transformer.encoder.fusion_layers.5.attn.v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.5.attn.v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.5.attn.l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.5.attn.l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.5.attn.values_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.5.attn.values_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.5.attn.values_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.5.attn.values_l_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.5.attn.out_v_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.5.attn.out_v_proj.bias
  ❄️ model.transformer.encoder.fusion_layers.5.attn.out_l_proj.weight
  ❄️ model.transformer.encoder.fusion_layers.5.attn.out_l_proj.bias
  ❄️ model.transformer.decoder.layers.0.cross_attn.sampling_offsets.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.cross_attn.sampling_offsets.base_layer.bias
  🔥 model.transformer.decoder.layers.0.cross_attn.sampling_offsets.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.cross_attn.sampling_offsets.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.cross_attn.attention_weights.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.cross_attn.attention_weights.base_layer.bias
  🔥 model.transformer.decoder.layers.0.cross_attn.attention_weights.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.cross_attn.attention_weights.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.cross_attn.value_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.cross_attn.value_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.0.cross_attn.value_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.cross_attn.value_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.cross_attn.output_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.cross_attn.output_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.0.cross_attn.output_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.cross_attn.output_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.norm1.weight
  ❄️ model.transformer.decoder.layers.0.norm1.bias
  ❄️ model.transformer.decoder.layers.0.ca_text.in_proj_weight
  ❄️ model.transformer.decoder.layers.0.ca_text.in_proj_bias
  ❄️ model.transformer.decoder.layers.0.ca_text.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.ca_text.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.0.ca_text.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.ca_text.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.catext_norm.weight
  ❄️ model.transformer.decoder.layers.0.catext_norm.bias
  ❄️ model.transformer.decoder.layers.0.self_attn.in_proj_weight
  ❄️ model.transformer.decoder.layers.0.self_attn.in_proj_bias
  ❄️ model.transformer.decoder.layers.0.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.0.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.norm2.weight
  ❄️ model.transformer.decoder.layers.0.norm2.bias
  ❄️ model.transformer.decoder.layers.0.linear1.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.linear1.base_layer.bias
  🔥 model.transformer.decoder.layers.0.linear1.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.linear1.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.linear2.base_layer.weight
  ❄️ model.transformer.decoder.layers.0.linear2.base_layer.bias
  🔥 model.transformer.decoder.layers.0.linear2.lora_A.default.weight
  🔥 model.transformer.decoder.layers.0.linear2.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.0.norm3.weight
  ❄️ model.transformer.decoder.layers.0.norm3.bias
  ❄️ model.transformer.decoder.layers.1.cross_attn.sampling_offsets.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.cross_attn.sampling_offsets.base_layer.bias
  🔥 model.transformer.decoder.layers.1.cross_attn.sampling_offsets.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.cross_attn.sampling_offsets.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.cross_attn.attention_weights.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.cross_attn.attention_weights.base_layer.bias
  🔥 model.transformer.decoder.layers.1.cross_attn.attention_weights.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.cross_attn.attention_weights.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.cross_attn.value_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.cross_attn.value_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.1.cross_attn.value_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.cross_attn.value_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.cross_attn.output_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.cross_attn.output_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.1.cross_attn.output_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.cross_attn.output_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.norm1.weight
  ❄️ model.transformer.decoder.layers.1.norm1.bias
  ❄️ model.transformer.decoder.layers.1.ca_text.in_proj_weight
  ❄️ model.transformer.decoder.layers.1.ca_text.in_proj_bias
  ❄️ model.transformer.decoder.layers.1.ca_text.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.ca_text.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.1.ca_text.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.ca_text.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.catext_norm.weight
  ❄️ model.transformer.decoder.layers.1.catext_norm.bias
  ❄️ model.transformer.decoder.layers.1.self_attn.in_proj_weight
  ❄️ model.transformer.decoder.layers.1.self_attn.in_proj_bias
  ❄️ model.transformer.decoder.layers.1.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.1.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.norm2.weight
  ❄️ model.transformer.decoder.layers.1.norm2.bias
  ❄️ model.transformer.decoder.layers.1.linear1.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.linear1.base_layer.bias
  🔥 model.transformer.decoder.layers.1.linear1.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.linear1.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.linear2.base_layer.weight
  ❄️ model.transformer.decoder.layers.1.linear2.base_layer.bias
  🔥 model.transformer.decoder.layers.1.linear2.lora_A.default.weight
  🔥 model.transformer.decoder.layers.1.linear2.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.1.norm3.weight
  ❄️ model.transformer.decoder.layers.1.norm3.bias
  ❄️ model.transformer.decoder.layers.2.cross_attn.sampling_offsets.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.cross_attn.sampling_offsets.base_layer.bias
  🔥 model.transformer.decoder.layers.2.cross_attn.sampling_offsets.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.cross_attn.sampling_offsets.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.cross_attn.attention_weights.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.cross_attn.attention_weights.base_layer.bias
  🔥 model.transformer.decoder.layers.2.cross_attn.attention_weights.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.cross_attn.attention_weights.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.cross_attn.value_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.cross_attn.value_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.2.cross_attn.value_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.cross_attn.value_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.cross_attn.output_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.cross_attn.output_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.2.cross_attn.output_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.cross_attn.output_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.norm1.weight
  ❄️ model.transformer.decoder.layers.2.norm1.bias
  ❄️ model.transformer.decoder.layers.2.ca_text.in_proj_weight
  ❄️ model.transformer.decoder.layers.2.ca_text.in_proj_bias
  ❄️ model.transformer.decoder.layers.2.ca_text.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.ca_text.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.2.ca_text.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.ca_text.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.catext_norm.weight
  ❄️ model.transformer.decoder.layers.2.catext_norm.bias
  ❄️ model.transformer.decoder.layers.2.self_attn.in_proj_weight
  ❄️ model.transformer.decoder.layers.2.self_attn.in_proj_bias
  ❄️ model.transformer.decoder.layers.2.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.2.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.norm2.weight
  ❄️ model.transformer.decoder.layers.2.norm2.bias
  ❄️ model.transformer.decoder.layers.2.linear1.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.linear1.base_layer.bias
  🔥 model.transformer.decoder.layers.2.linear1.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.linear1.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.linear2.base_layer.weight
  ❄️ model.transformer.decoder.layers.2.linear2.base_layer.bias
  🔥 model.transformer.decoder.layers.2.linear2.lora_A.default.weight
  🔥 model.transformer.decoder.layers.2.linear2.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.2.norm3.weight
  ❄️ model.transformer.decoder.layers.2.norm3.bias
  ❄️ model.transformer.decoder.layers.3.cross_attn.sampling_offsets.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.cross_attn.sampling_offsets.base_layer.bias
  🔥 model.transformer.decoder.layers.3.cross_attn.sampling_offsets.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.cross_attn.sampling_offsets.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.cross_attn.attention_weights.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.cross_attn.attention_weights.base_layer.bias
  🔥 model.transformer.decoder.layers.3.cross_attn.attention_weights.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.cross_attn.attention_weights.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.cross_attn.value_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.cross_attn.value_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.3.cross_attn.value_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.cross_attn.value_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.cross_attn.output_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.cross_attn.output_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.3.cross_attn.output_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.cross_attn.output_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.norm1.weight
  ❄️ model.transformer.decoder.layers.3.norm1.bias
  ❄️ model.transformer.decoder.layers.3.ca_text.in_proj_weight
  ❄️ model.transformer.decoder.layers.3.ca_text.in_proj_bias
  ❄️ model.transformer.decoder.layers.3.ca_text.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.ca_text.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.3.ca_text.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.ca_text.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.catext_norm.weight
  ❄️ model.transformer.decoder.layers.3.catext_norm.bias
  ❄️ model.transformer.decoder.layers.3.self_attn.in_proj_weight
  ❄️ model.transformer.decoder.layers.3.self_attn.in_proj_bias
  ❄️ model.transformer.decoder.layers.3.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.3.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.norm2.weight
  ❄️ model.transformer.decoder.layers.3.norm2.bias
  ❄️ model.transformer.decoder.layers.3.linear1.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.linear1.base_layer.bias
  🔥 model.transformer.decoder.layers.3.linear1.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.linear1.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.linear2.base_layer.weight
  ❄️ model.transformer.decoder.layers.3.linear2.base_layer.bias
  🔥 model.transformer.decoder.layers.3.linear2.lora_A.default.weight
  🔥 model.transformer.decoder.layers.3.linear2.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.3.norm3.weight
  ❄️ model.transformer.decoder.layers.3.norm3.bias
  ❄️ model.transformer.decoder.layers.4.cross_attn.sampling_offsets.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.cross_attn.sampling_offsets.base_layer.bias
  🔥 model.transformer.decoder.layers.4.cross_attn.sampling_offsets.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.cross_attn.sampling_offsets.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.cross_attn.attention_weights.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.cross_attn.attention_weights.base_layer.bias
  🔥 model.transformer.decoder.layers.4.cross_attn.attention_weights.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.cross_attn.attention_weights.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.cross_attn.value_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.cross_attn.value_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.4.cross_attn.value_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.cross_attn.value_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.cross_attn.output_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.cross_attn.output_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.4.cross_attn.output_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.cross_attn.output_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.norm1.weight
  ❄️ model.transformer.decoder.layers.4.norm1.bias
  ❄️ model.transformer.decoder.layers.4.ca_text.in_proj_weight
  ❄️ model.transformer.decoder.layers.4.ca_text.in_proj_bias
  ❄️ model.transformer.decoder.layers.4.ca_text.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.ca_text.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.4.ca_text.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.ca_text.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.catext_norm.weight
  ❄️ model.transformer.decoder.layers.4.catext_norm.bias
  ❄️ model.transformer.decoder.layers.4.self_attn.in_proj_weight
  ❄️ model.transformer.decoder.layers.4.self_attn.in_proj_bias
  ❄️ model.transformer.decoder.layers.4.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.4.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.norm2.weight
  ❄️ model.transformer.decoder.layers.4.norm2.bias
  ❄️ model.transformer.decoder.layers.4.linear1.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.linear1.base_layer.bias
  🔥 model.transformer.decoder.layers.4.linear1.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.linear1.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.linear2.base_layer.weight
  ❄️ model.transformer.decoder.layers.4.linear2.base_layer.bias
  🔥 model.transformer.decoder.layers.4.linear2.lora_A.default.weight
  🔥 model.transformer.decoder.layers.4.linear2.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.4.norm3.weight
  ❄️ model.transformer.decoder.layers.4.norm3.bias
  ❄️ model.transformer.decoder.layers.5.cross_attn.sampling_offsets.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.cross_attn.sampling_offsets.base_layer.bias
  🔥 model.transformer.decoder.layers.5.cross_attn.sampling_offsets.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.cross_attn.sampling_offsets.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.cross_attn.attention_weights.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.cross_attn.attention_weights.base_layer.bias
  🔥 model.transformer.decoder.layers.5.cross_attn.attention_weights.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.cross_attn.attention_weights.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.cross_attn.value_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.cross_attn.value_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.5.cross_attn.value_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.cross_attn.value_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.cross_attn.output_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.cross_attn.output_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.5.cross_attn.output_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.cross_attn.output_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.norm1.weight
  ❄️ model.transformer.decoder.layers.5.norm1.bias
  ❄️ model.transformer.decoder.layers.5.ca_text.in_proj_weight
  ❄️ model.transformer.decoder.layers.5.ca_text.in_proj_bias
  ❄️ model.transformer.decoder.layers.5.ca_text.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.ca_text.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.5.ca_text.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.ca_text.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.catext_norm.weight
  ❄️ model.transformer.decoder.layers.5.catext_norm.bias
  ❄️ model.transformer.decoder.layers.5.self_attn.in_proj_weight
  ❄️ model.transformer.decoder.layers.5.self_attn.in_proj_bias
  ❄️ model.transformer.decoder.layers.5.self_attn.out_proj.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.self_attn.out_proj.base_layer.bias
  🔥 model.transformer.decoder.layers.5.self_attn.out_proj.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.self_attn.out_proj.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.norm2.weight
  ❄️ model.transformer.decoder.layers.5.norm2.bias
  ❄️ model.transformer.decoder.layers.5.linear1.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.linear1.base_layer.bias
  🔥 model.transformer.decoder.layers.5.linear1.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.linear1.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.linear2.base_layer.weight
  ❄️ model.transformer.decoder.layers.5.linear2.base_layer.bias
  🔥 model.transformer.decoder.layers.5.linear2.lora_A.default.weight
  🔥 model.transformer.decoder.layers.5.linear2.lora_B.default.weight
  ❄️ model.transformer.decoder.layers.5.norm3.weight
  ❄️ model.transformer.decoder.layers.5.norm3.bias
  ❄️ model.transformer.decoder.norm.weight
  ❄️ model.transformer.decoder.norm.bias
  ❄️ model.transformer.decoder.ref_point_head.layers.0.weight
  ❄️ model.transformer.decoder.ref_point_head.layers.0.bias
  ❄️ model.transformer.decoder.ref_point_head.layers.1.weight
  ❄️ model.transformer.decoder.ref_point_head.layers.1.bias
  ❄️ model.transformer.decoder.bbox_embed.0.layers.0.base_layer.weight
  ❄️ model.transformer.decoder.bbox_embed.0.layers.0.base_layer.bias
  🔥 model.transformer.decoder.bbox_embed.0.layers.0.lora_A.default.weight
  🔥 model.transformer.decoder.bbox_embed.0.layers.0.lora_B.default.weight
  ❄️ model.transformer.decoder.bbox_embed.0.layers.1.base_layer.weight
  ❄️ model.transformer.decoder.bbox_embed.0.layers.1.base_layer.bias
  🔥 model.transformer.decoder.bbox_embed.0.layers.1.lora_A.default.weight
  🔥 model.transformer.decoder.bbox_embed.0.layers.1.lora_B.default.weight
  ❄️ model.transformer.decoder.bbox_embed.0.layers.2.original_module.weight
  ❄️ model.transformer.decoder.bbox_embed.0.layers.2.original_module.bias
  🔥 model.transformer.decoder.bbox_embed.0.layers.2.modules_to_save.default.weight
  🔥 model.transformer.decoder.bbox_embed.0.layers.2.modules_to_save.default.bias
  ❄️ model.transformer.tgt_embed.weight
  ❄️ model.transformer.enc_output.weight
  ❄️ model.transformer.enc_output.bias
  ❄️ model.transformer.enc_output_norm.weight
  ❄️ model.transformer.enc_output_norm.bias
  ❄️ model.transformer.enc_out_bbox_embed.layers.0.weight
  ❄️ model.transformer.enc_out_bbox_embed.layers.0.bias
  ❄️ model.transformer.enc_out_bbox_embed.layers.1.weight
  ❄️ model.transformer.enc_out_bbox_embed.layers.1.bias
  ❄️ model.transformer.enc_out_bbox_embed.layers.2.weight
  ❄️ model.transformer.enc_out_bbox_embed.layers.2.bias
  ❄️ model.bert.embeddings.word_embeddings.weight
  ❄️ model.bert.embeddings.position_embeddings.weight
  ❄️ model.bert.embeddings.token_type_embeddings.weight
  ❄️ model.bert.embeddings.LayerNorm.weight
  ❄️ model.bert.embeddings.LayerNorm.bias
  ❄️ model.bert.encoder.layer.0.attention.self.query.weight
  ❄️ model.bert.encoder.layer.0.attention.self.query.bias
  ❄️ model.bert.encoder.layer.0.attention.self.key.weight
  ❄️ model.bert.encoder.layer.0.attention.self.key.bias
  ❄️ model.bert.encoder.layer.0.attention.self.value.weight
  ❄️ model.bert.encoder.layer.0.attention.self.value.bias
  ❄️ model.bert.encoder.layer.0.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.0.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.0.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.0.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.0.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.0.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.0.output.dense.weight
  ❄️ model.bert.encoder.layer.0.output.dense.bias
  ❄️ model.bert.encoder.layer.0.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.0.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.1.attention.self.query.weight
  ❄️ model.bert.encoder.layer.1.attention.self.query.bias
  ❄️ model.bert.encoder.layer.1.attention.self.key.weight
  ❄️ model.bert.encoder.layer.1.attention.self.key.bias
  ❄️ model.bert.encoder.layer.1.attention.self.value.weight
  ❄️ model.bert.encoder.layer.1.attention.self.value.bias
  ❄️ model.bert.encoder.layer.1.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.1.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.1.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.1.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.1.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.1.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.1.output.dense.weight
  ❄️ model.bert.encoder.layer.1.output.dense.bias
  ❄️ model.bert.encoder.layer.1.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.1.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.2.attention.self.query.weight
  ❄️ model.bert.encoder.layer.2.attention.self.query.bias
  ❄️ model.bert.encoder.layer.2.attention.self.key.weight
  ❄️ model.bert.encoder.layer.2.attention.self.key.bias
  ❄️ model.bert.encoder.layer.2.attention.self.value.weight
  ❄️ model.bert.encoder.layer.2.attention.self.value.bias
  ❄️ model.bert.encoder.layer.2.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.2.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.2.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.2.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.2.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.2.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.2.output.dense.weight
  ❄️ model.bert.encoder.layer.2.output.dense.bias
  ❄️ model.bert.encoder.layer.2.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.2.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.3.attention.self.query.weight
  ❄️ model.bert.encoder.layer.3.attention.self.query.bias
  ❄️ model.bert.encoder.layer.3.attention.self.key.weight
  ❄️ model.bert.encoder.layer.3.attention.self.key.bias
  ❄️ model.bert.encoder.layer.3.attention.self.value.weight
  ❄️ model.bert.encoder.layer.3.attention.self.value.bias
  ❄️ model.bert.encoder.layer.3.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.3.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.3.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.3.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.3.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.3.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.3.output.dense.weight
  ❄️ model.bert.encoder.layer.3.output.dense.bias
  ❄️ model.bert.encoder.layer.3.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.3.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.4.attention.self.query.weight
  ❄️ model.bert.encoder.layer.4.attention.self.query.bias
  ❄️ model.bert.encoder.layer.4.attention.self.key.weight
  ❄️ model.bert.encoder.layer.4.attention.self.key.bias
  ❄️ model.bert.encoder.layer.4.attention.self.value.weight
  ❄️ model.bert.encoder.layer.4.attention.self.value.bias
  ❄️ model.bert.encoder.layer.4.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.4.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.4.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.4.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.4.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.4.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.4.output.dense.weight
  ❄️ model.bert.encoder.layer.4.output.dense.bias
  ❄️ model.bert.encoder.layer.4.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.4.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.5.attention.self.query.weight
  ❄️ model.bert.encoder.layer.5.attention.self.query.bias
  ❄️ model.bert.encoder.layer.5.attention.self.key.weight
  ❄️ model.bert.encoder.layer.5.attention.self.key.bias
  ❄️ model.bert.encoder.layer.5.attention.self.value.weight
  ❄️ model.bert.encoder.layer.5.attention.self.value.bias
  ❄️ model.bert.encoder.layer.5.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.5.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.5.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.5.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.5.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.5.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.5.output.dense.weight
  ❄️ model.bert.encoder.layer.5.output.dense.bias
  ❄️ model.bert.encoder.layer.5.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.5.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.6.attention.self.query.weight
  ❄️ model.bert.encoder.layer.6.attention.self.query.bias
  ❄️ model.bert.encoder.layer.6.attention.self.key.weight
  ❄️ model.bert.encoder.layer.6.attention.self.key.bias
  ❄️ model.bert.encoder.layer.6.attention.self.value.weight
  ❄️ model.bert.encoder.layer.6.attention.self.value.bias
  ❄️ model.bert.encoder.layer.6.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.6.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.6.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.6.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.6.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.6.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.6.output.dense.weight
  ❄️ model.bert.encoder.layer.6.output.dense.bias
  ❄️ model.bert.encoder.layer.6.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.6.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.7.attention.self.query.weight
  ❄️ model.bert.encoder.layer.7.attention.self.query.bias
  ❄️ model.bert.encoder.layer.7.attention.self.key.weight
  ❄️ model.bert.encoder.layer.7.attention.self.key.bias
  ❄️ model.bert.encoder.layer.7.attention.self.value.weight
  ❄️ model.bert.encoder.layer.7.attention.self.value.bias
  ❄️ model.bert.encoder.layer.7.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.7.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.7.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.7.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.7.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.7.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.7.output.dense.weight
  ❄️ model.bert.encoder.layer.7.output.dense.bias
  ❄️ model.bert.encoder.layer.7.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.7.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.8.attention.self.query.weight
  ❄️ model.bert.encoder.layer.8.attention.self.query.bias
  ❄️ model.bert.encoder.layer.8.attention.self.key.weight
  ❄️ model.bert.encoder.layer.8.attention.self.key.bias
  ❄️ model.bert.encoder.layer.8.attention.self.value.weight
  ❄️ model.bert.encoder.layer.8.attention.self.value.bias
  ❄️ model.bert.encoder.layer.8.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.8.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.8.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.8.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.8.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.8.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.8.output.dense.weight
  ❄️ model.bert.encoder.layer.8.output.dense.bias
  ❄️ model.bert.encoder.layer.8.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.8.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.9.attention.self.query.weight
  ❄️ model.bert.encoder.layer.9.attention.self.query.bias
  ❄️ model.bert.encoder.layer.9.attention.self.key.weight
  ❄️ model.bert.encoder.layer.9.attention.self.key.bias
  ❄️ model.bert.encoder.layer.9.attention.self.value.weight
  ❄️ model.bert.encoder.layer.9.attention.self.value.bias
  ❄️ model.bert.encoder.layer.9.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.9.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.9.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.9.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.9.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.9.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.9.output.dense.weight
  ❄️ model.bert.encoder.layer.9.output.dense.bias
  ❄️ model.bert.encoder.layer.9.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.9.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.10.attention.self.query.weight
  ❄️ model.bert.encoder.layer.10.attention.self.query.bias
  ❄️ model.bert.encoder.layer.10.attention.self.key.weight
  ❄️ model.bert.encoder.layer.10.attention.self.key.bias
  ❄️ model.bert.encoder.layer.10.attention.self.value.weight
  ❄️ model.bert.encoder.layer.10.attention.self.value.bias
  ❄️ model.bert.encoder.layer.10.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.10.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.10.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.10.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.10.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.10.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.10.output.dense.weight
  ❄️ model.bert.encoder.layer.10.output.dense.bias
  ❄️ model.bert.encoder.layer.10.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.10.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.11.attention.self.query.weight
  ❄️ model.bert.encoder.layer.11.attention.self.query.bias
  ❄️ model.bert.encoder.layer.11.attention.self.key.weight
  ❄️ model.bert.encoder.layer.11.attention.self.key.bias
  ❄️ model.bert.encoder.layer.11.attention.self.value.weight
  ❄️ model.bert.encoder.layer.11.attention.self.value.bias
  ❄️ model.bert.encoder.layer.11.attention.output.dense.weight
  ❄️ model.bert.encoder.layer.11.attention.output.dense.bias
  ❄️ model.bert.encoder.layer.11.attention.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.11.attention.output.LayerNorm.bias
  ❄️ model.bert.encoder.layer.11.intermediate.dense.weight
  ❄️ model.bert.encoder.layer.11.intermediate.dense.bias
  ❄️ model.bert.encoder.layer.11.output.dense.weight
  ❄️ model.bert.encoder.layer.11.output.dense.bias
  ❄️ model.bert.encoder.layer.11.output.LayerNorm.weight
  ❄️ model.bert.encoder.layer.11.output.LayerNorm.bias
  ❄️ model.bert.pooler.dense.weight
  ❄️ model.bert.pooler.dense.bias
  ❄️ model.feat_map.base_layer.weight
  ❄️ model.feat_map.base_layer.bias
  🔥 model.feat_map.lora_A.default.weight
  🔥 model.feat_map.lora_B.default.weight
  ❄️ model.input_proj.0.0.weight
  ❄️ model.input_proj.0.0.bias
  ❄️ model.input_proj.0.1.weight
  ❄️ model.input_proj.0.1.bias
  ❄️ model.input_proj.1.0.weight
  ❄️ model.input_proj.1.0.bias
  ❄️ model.input_proj.1.1.weight
  ❄️ model.input_proj.1.1.bias
  ❄️ model.input_proj.2.0.weight
  ❄️ model.input_proj.2.0.bias
  ❄️ model.input_proj.2.1.weight
  ❄️ model.input_proj.2.1.bias
  ❄️ model.input_proj.3.0.weight
  ❄️ model.input_proj.3.0.bias
  ❄️ model.input_proj.3.1.weight
  ❄️ model.input_proj.3.1.bias
  ❄️ model.backbone.0.patch_embed.proj.weight
  ❄️ model.backbone.0.patch_embed.proj.bias
  ❄️ model.backbone.0.patch_embed.norm.weight
  ❄️ model.backbone.0.patch_embed.norm.bias
  ❄️ model.backbone.0.layers.0.blocks.0.norm1.weight
  ❄️ model.backbone.0.layers.0.blocks.0.norm1.bias
  ❄️ model.backbone.0.layers.0.blocks.0.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.0.blocks.0.attn.qkv.weight
  ❄️ model.backbone.0.layers.0.blocks.0.attn.qkv.bias
  ❄️ model.backbone.0.layers.0.blocks.0.attn.proj.weight
  ❄️ model.backbone.0.layers.0.blocks.0.attn.proj.bias
  ❄️ model.backbone.0.layers.0.blocks.0.norm2.weight
  ❄️ model.backbone.0.layers.0.blocks.0.norm2.bias
  ❄️ model.backbone.0.layers.0.blocks.0.mlp.fc1.weight
  ❄️ model.backbone.0.layers.0.blocks.0.mlp.fc1.bias
  ❄️ model.backbone.0.layers.0.blocks.0.mlp.fc2.weight
  ❄️ model.backbone.0.layers.0.blocks.0.mlp.fc2.bias
  ❄️ model.backbone.0.layers.0.blocks.1.norm1.weight
  ❄️ model.backbone.0.layers.0.blocks.1.norm1.bias
  ❄️ model.backbone.0.layers.0.blocks.1.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.0.blocks.1.attn.qkv.weight
  ❄️ model.backbone.0.layers.0.blocks.1.attn.qkv.bias
  ❄️ model.backbone.0.layers.0.blocks.1.attn.proj.weight
  ❄️ model.backbone.0.layers.0.blocks.1.attn.proj.bias
  ❄️ model.backbone.0.layers.0.blocks.1.norm2.weight
  ❄️ model.backbone.0.layers.0.blocks.1.norm2.bias
  ❄️ model.backbone.0.layers.0.blocks.1.mlp.fc1.weight
  ❄️ model.backbone.0.layers.0.blocks.1.mlp.fc1.bias
  ❄️ model.backbone.0.layers.0.blocks.1.mlp.fc2.weight
  ❄️ model.backbone.0.layers.0.blocks.1.mlp.fc2.bias
  ❄️ model.backbone.0.layers.0.downsample.reduction.weight
  ❄️ model.backbone.0.layers.0.downsample.norm.weight
  ❄️ model.backbone.0.layers.0.downsample.norm.bias
  ❄️ model.backbone.0.layers.1.blocks.0.norm1.weight
  ❄️ model.backbone.0.layers.1.blocks.0.norm1.bias
  ❄️ model.backbone.0.layers.1.blocks.0.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.1.blocks.0.attn.qkv.weight
  ❄️ model.backbone.0.layers.1.blocks.0.attn.qkv.bias
  ❄️ model.backbone.0.layers.1.blocks.0.attn.proj.weight
  ❄️ model.backbone.0.layers.1.blocks.0.attn.proj.bias
  ❄️ model.backbone.0.layers.1.blocks.0.norm2.weight
  ❄️ model.backbone.0.layers.1.blocks.0.norm2.bias
  ❄️ model.backbone.0.layers.1.blocks.0.mlp.fc1.weight
  ❄️ model.backbone.0.layers.1.blocks.0.mlp.fc1.bias
  ❄️ model.backbone.0.layers.1.blocks.0.mlp.fc2.weight
  ❄️ model.backbone.0.layers.1.blocks.0.mlp.fc2.bias
  ❄️ model.backbone.0.layers.1.blocks.1.norm1.weight
  ❄️ model.backbone.0.layers.1.blocks.1.norm1.bias
  ❄️ model.backbone.0.layers.1.blocks.1.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.1.blocks.1.attn.qkv.weight
  ❄️ model.backbone.0.layers.1.blocks.1.attn.qkv.bias
  ❄️ model.backbone.0.layers.1.blocks.1.attn.proj.weight
  ❄️ model.backbone.0.layers.1.blocks.1.attn.proj.bias
  ❄️ model.backbone.0.layers.1.blocks.1.norm2.weight
  ❄️ model.backbone.0.layers.1.blocks.1.norm2.bias
  ❄️ model.backbone.0.layers.1.blocks.1.mlp.fc1.weight
  ❄️ model.backbone.0.layers.1.blocks.1.mlp.fc1.bias
  ❄️ model.backbone.0.layers.1.blocks.1.mlp.fc2.weight
  ❄️ model.backbone.0.layers.1.blocks.1.mlp.fc2.bias
  ❄️ model.backbone.0.layers.1.downsample.reduction.weight
  ❄️ model.backbone.0.layers.1.downsample.norm.weight
  ❄️ model.backbone.0.layers.1.downsample.norm.bias
  ❄️ model.backbone.0.layers.2.blocks.0.norm1.weight
  ❄️ model.backbone.0.layers.2.blocks.0.norm1.bias
  ❄️ model.backbone.0.layers.2.blocks.0.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.2.blocks.0.attn.qkv.weight
  ❄️ model.backbone.0.layers.2.blocks.0.attn.qkv.bias
  ❄️ model.backbone.0.layers.2.blocks.0.attn.proj.weight
  ❄️ model.backbone.0.layers.2.blocks.0.attn.proj.bias
  ❄️ model.backbone.0.layers.2.blocks.0.norm2.weight
  ❄️ model.backbone.0.layers.2.blocks.0.norm2.bias
  ❄️ model.backbone.0.layers.2.blocks.0.mlp.fc1.weight
  ❄️ model.backbone.0.layers.2.blocks.0.mlp.fc1.bias
  ❄️ model.backbone.0.layers.2.blocks.0.mlp.fc2.weight
  ❄️ model.backbone.0.layers.2.blocks.0.mlp.fc2.bias
  ❄️ model.backbone.0.layers.2.blocks.1.norm1.weight
  ❄️ model.backbone.0.layers.2.blocks.1.norm1.bias
  ❄️ model.backbone.0.layers.2.blocks.1.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.2.blocks.1.attn.qkv.weight
  ❄️ model.backbone.0.layers.2.blocks.1.attn.qkv.bias
  ❄️ model.backbone.0.layers.2.blocks.1.attn.proj.weight
  ❄️ model.backbone.0.layers.2.blocks.1.attn.proj.bias
  ❄️ model.backbone.0.layers.2.blocks.1.norm2.weight
  ❄️ model.backbone.0.layers.2.blocks.1.norm2.bias
  ❄️ model.backbone.0.layers.2.blocks.1.mlp.fc1.weight
  ❄️ model.backbone.0.layers.2.blocks.1.mlp.fc1.bias
  ❄️ model.backbone.0.layers.2.blocks.1.mlp.fc2.weight
  ❄️ model.backbone.0.layers.2.blocks.1.mlp.fc2.bias
  ❄️ model.backbone.0.layers.2.blocks.2.norm1.weight
  ❄️ model.backbone.0.layers.2.blocks.2.norm1.bias
  ❄️ model.backbone.0.layers.2.blocks.2.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.2.blocks.2.attn.qkv.weight
  ❄️ model.backbone.0.layers.2.blocks.2.attn.qkv.bias
  ❄️ model.backbone.0.layers.2.blocks.2.attn.proj.weight
  ❄️ model.backbone.0.layers.2.blocks.2.attn.proj.bias
  ❄️ model.backbone.0.layers.2.blocks.2.norm2.weight
  ❄️ model.backbone.0.layers.2.blocks.2.norm2.bias
  ❄️ model.backbone.0.layers.2.blocks.2.mlp.fc1.weight
  ❄️ model.backbone.0.layers.2.blocks.2.mlp.fc1.bias
  ❄️ model.backbone.0.layers.2.blocks.2.mlp.fc2.weight
  ❄️ model.backbone.0.layers.2.blocks.2.mlp.fc2.bias
  ❄️ model.backbone.0.layers.2.blocks.3.norm1.weight
  ❄️ model.backbone.0.layers.2.blocks.3.norm1.bias
  ❄️ model.backbone.0.layers.2.blocks.3.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.2.blocks.3.attn.qkv.weight
  ❄️ model.backbone.0.layers.2.blocks.3.attn.qkv.bias
  ❄️ model.backbone.0.layers.2.blocks.3.attn.proj.weight
  ❄️ model.backbone.0.layers.2.blocks.3.attn.proj.bias
  ❄️ model.backbone.0.layers.2.blocks.3.norm2.weight
  ❄️ model.backbone.0.layers.2.blocks.3.norm2.bias
  ❄️ model.backbone.0.layers.2.blocks.3.mlp.fc1.weight
  ❄️ model.backbone.0.layers.2.blocks.3.mlp.fc1.bias
  ❄️ model.backbone.0.layers.2.blocks.3.mlp.fc2.weight
  ❄️ model.backbone.0.layers.2.blocks.3.mlp.fc2.bias
  ❄️ model.backbone.0.layers.2.blocks.4.norm1.weight
  ❄️ model.backbone.0.layers.2.blocks.4.norm1.bias
  ❄️ model.backbone.0.layers.2.blocks.4.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.2.blocks.4.attn.qkv.weight
  ❄️ model.backbone.0.layers.2.blocks.4.attn.qkv.bias
  ❄️ model.backbone.0.layers.2.blocks.4.attn.proj.weight
  ❄️ model.backbone.0.layers.2.blocks.4.attn.proj.bias
  ❄️ model.backbone.0.layers.2.blocks.4.norm2.weight
  ❄️ model.backbone.0.layers.2.blocks.4.norm2.bias
  ❄️ model.backbone.0.layers.2.blocks.4.mlp.fc1.weight
  ❄️ model.backbone.0.layers.2.blocks.4.mlp.fc1.bias
  ❄️ model.backbone.0.layers.2.blocks.4.mlp.fc2.weight
  ❄️ model.backbone.0.layers.2.blocks.4.mlp.fc2.bias
  ❄️ model.backbone.0.layers.2.blocks.5.norm1.weight
  ❄️ model.backbone.0.layers.2.blocks.5.norm1.bias
  ❄️ model.backbone.0.layers.2.blocks.5.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.2.blocks.5.attn.qkv.weight
  ❄️ model.backbone.0.layers.2.blocks.5.attn.qkv.bias
  ❄️ model.backbone.0.layers.2.blocks.5.attn.proj.weight
  ❄️ model.backbone.0.layers.2.blocks.5.attn.proj.bias
  ❄️ model.backbone.0.layers.2.blocks.5.norm2.weight
  ❄️ model.backbone.0.layers.2.blocks.5.norm2.bias
  ❄️ model.backbone.0.layers.2.blocks.5.mlp.fc1.weight
  ❄️ model.backbone.0.layers.2.blocks.5.mlp.fc1.bias
  ❄️ model.backbone.0.layers.2.blocks.5.mlp.fc2.weight
  ❄️ model.backbone.0.layers.2.blocks.5.mlp.fc2.bias
  ❄️ model.backbone.0.layers.2.downsample.reduction.weight
  ❄️ model.backbone.0.layers.2.downsample.norm.weight
  ❄️ model.backbone.0.layers.2.downsample.norm.bias
  ❄️ model.backbone.0.layers.3.blocks.0.norm1.weight
  ❄️ model.backbone.0.layers.3.blocks.0.norm1.bias
  ❄️ model.backbone.0.layers.3.blocks.0.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.3.blocks.0.attn.qkv.weight
  ❄️ model.backbone.0.layers.3.blocks.0.attn.qkv.bias
  ❄️ model.backbone.0.layers.3.blocks.0.attn.proj.weight
  ❄️ model.backbone.0.layers.3.blocks.0.attn.proj.bias
  ❄️ model.backbone.0.layers.3.blocks.0.norm2.weight
  ❄️ model.backbone.0.layers.3.blocks.0.norm2.bias
  ❄️ model.backbone.0.layers.3.blocks.0.mlp.fc1.weight
  ❄️ model.backbone.0.layers.3.blocks.0.mlp.fc1.bias
  ❄️ model.backbone.0.layers.3.blocks.0.mlp.fc2.weight
  ❄️ model.backbone.0.layers.3.blocks.0.mlp.fc2.bias
  ❄️ model.backbone.0.layers.3.blocks.1.norm1.weight
  ❄️ model.backbone.0.layers.3.blocks.1.norm1.bias
  ❄️ model.backbone.0.layers.3.blocks.1.attn.relative_position_bias_table
  ❄️ model.backbone.0.layers.3.blocks.1.attn.qkv.weight
  ❄️ model.backbone.0.layers.3.blocks.1.attn.qkv.bias
  ❄️ model.backbone.0.layers.3.blocks.1.attn.proj.weight
  ❄️ model.backbone.0.layers.3.blocks.1.attn.proj.bias
  ❄️ model.backbone.0.layers.3.blocks.1.norm2.weight
  ❄️ model.backbone.0.layers.3.blocks.1.norm2.bias
  ❄️ model.backbone.0.layers.3.blocks.1.mlp.fc1.weight
  ❄️ model.backbone.0.layers.3.blocks.1.mlp.fc1.bias
  ❄️ model.backbone.0.layers.3.blocks.1.mlp.fc2.weight
  ❄️ model.backbone.0.layers.3.blocks.1.mlp.fc2.bias
  ❄️ model.backbone.0.norm1.weight
  ❄️ model.backbone.0.norm1.bias
  ❄️ model.backbone.0.norm2.weight
  ❄️ model.backbone.0.norm2.bias
  ❄️ model.backbone.0.norm3.weight
  ❄️ model.backbone.0.norm3.bias
  Frozen parameters: 172,839,682
  Trainable parameters: 2,991,108

=== Overall Status ===
Total Parameters: 175,830,790
Frozen Parameters: 172,839,682 (98.30%)
Trainable Parameters: 2,991,108 (1.70%)
/home/hamze/anaconda3/envs/groundingdino/lib/python3.10/site-packages/transformers/modeling_utils.py:1575: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.
  warnings.warn(
/home/hamze/anaconda3/envs/groundingdino/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:838: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  return fn(*args, **kwargs)
/home/hamze/anaconda3/envs/groundingdino/lib/python3.10/site-packages/torch/utils/checkpoint.py:86: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
/home/hamze/Documents/Grounding-Sam-Ultrasound/groundingdino/models/GroundingDINO/transformer.py:862: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):