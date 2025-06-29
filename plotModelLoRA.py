
#%%
config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=[
            # Decoder cross attention
            "cross_attn.sampling_offsets",
            "cross_attn.attention_weights", 
            "cross_attn.value_proj",
            "cross_attn.output_proj",
            # Text cross attention
            "ca_text.out_proj",
            # Self attention 
            "self_attn.out_proj",
            # FFN
            "linear1",
            "linear2",
            # Bbox prediction layers
            "bbox_embed.0.layers.0",
            "bbox_embed.0.layers.1",
            # fearue map
            "feat_map"
        ],
        modules_to_save=["bbox_embed.0.layers.2"],
        bias="none",
        inference_mode=inference,
    )

#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%55

# /home/hamze/anaconda3/envs/groundingdino/lib/python3.10/site-packages/torch/functional.py:554: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4314.)
#   return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
# final text_encoder_type: bert-base-uncased
# Adding Lora to model!
# Converting model to LoRA...
# LoRA parameters: 2,991,108 / 175,830,790 = 1.70%
# Lora model is 

# WARNING: Found non-LoRA trainable parameters:
# - base_model.model.transformer.decoder.bbox_embed.0.layers.2.modules_to_save.default.weight
# - base_model.model.transformer.decoder.bbox_embed.0.layers.2.modules_to_save.default.bias
# Is only Lora trainable?  False 
PeftModel(
  (base_model): LoraModel(
    (model): GroundingDINO(
      (transformer): Transformer(
        (encoder): TransformerEncoder(
          (layers): ModuleList(
            (0-5): 6 x DeformableTransformerEncoderLayer(
              (self_attn): MultiScaleDeformableAttention(
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
              (dropout1): Dropout(p=0.0, inplace=False)
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (linear1): lora.Linear(
                (base_layer): Linear(in_features=256, out_features=2048, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=256, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (dropout2): Dropout(p=0.0, inplace=False)
              (linear2): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=256, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=256, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (dropout3): Dropout(p=0.0, inplace=False)
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (text_layers): ModuleList(
            (0-5): 6 x TransformerEncoderLayer(
              (self_attn): MultiheadAttention(
                (out_proj): lora.Linear(
                  (base_layer): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=256, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
              )
              (linear1): lora.Linear(
                (base_layer): Linear(in_features=256, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=256, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (dropout): Dropout(p=0.0, inplace=False)
              (linear2): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=256, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=256, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (dropout1): Dropout(p=0.0, inplace=False)
              (dropout2): Dropout(p=0.0, inplace=False)
            )
          )
          (fusion_layers): ModuleList(
            (0-5): 6 x BiAttentionBlock(
              (layer_norm_v): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (layer_norm_l): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (attn): BiMultiHeadAttention(
                (v_proj): Linear(in_features=256, out_features=1024, bias=True)
                (l_proj): Linear(in_features=256, out_features=1024, bias=True)
                (values_v_proj): Linear(in_features=256, out_features=1024, bias=True)
                (values_l_proj): Linear(in_features=256, out_features=1024, bias=True)
                (out_v_proj): Linear(in_features=1024, out_features=256, bias=True)
                (out_l_proj): Linear(in_features=1024, out_features=256, bias=True)
              )
              (drop_path): DropPath(drop_prob=0.100)
            )
          )
        )
        (decoder): TransformerDecoder(
          (layers): ModuleList(
            (0-5): 6 x DeformableTransformerDecoderLayer(
              (cross_attn): MultiScaleDeformableAttention(
                (sampling_offsets): lora.Linear(
                  (base_layer): Linear(in_features=256, out_features=256, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=256, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (attention_weights): lora.Linear(
                  (base_layer): Linear(in_features=256, out_features=128, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=128, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (value_proj): lora.Linear(
                  (base_layer): Linear(in_features=256, out_features=256, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=256, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (output_proj): lora.Linear(
                  (base_layer): Linear(in_features=256, out_features=256, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=256, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
              )
              (dropout1): Identity()
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (ca_text): MultiheadAttention(
                (out_proj): lora.Linear(
                  (base_layer): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=256, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
              )
              (catext_dropout): Identity()
              (catext_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (self_attn): MultiheadAttention(
                (out_proj): lora.Linear(
                  (base_layer): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=256, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
              )
              (dropout2): Identity()
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (linear1): lora.Linear(
                (base_layer): Linear(in_features=256, out_features=2048, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=256, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (dropout3): Identity()
              (linear2): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=256, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=256, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (dropout4): Identity()
              (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (ref_point_head): MLP(
            (layers): ModuleList(
              (0): Linear(in_features=512, out_features=256, bias=True)
              (1): Linear(in_features=256, out_features=256, bias=True)
            )
          )
          (bbox_embed): ModuleList(
            (0-5): 6 x MLP(
              (layers): ModuleList(
                (0-1): 2 x lora.Linear(
                  (base_layer): Linear(in_features=256, out_features=256, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Identity()
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=256, out_features=32, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=32, out_features=256, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (2): ModulesToSaveWrapper(
                  (original_module): Linear(in_features=256, out_features=4, bias=True)
                  (modules_to_save): ModuleDict(
                    (default): Linear(in_features=256, out_features=4, bias=True)
                  )
                )
              )
            )
          )
          (class_embed): ModuleList(
            (0-5): 6 x ContrastiveEmbed()
          )
        )
        (tgt_embed): Embedding(900, 256)
        (enc_output): Linear(in_features=256, out_features=256, bias=True)
        (enc_output_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (enc_out_bbox_embed): MLP(
          (layers): ModuleList(
            (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
            (2): Linear(in_features=256, out_features=4, bias=True)
          )
        )
        (enc_out_class_embed): ContrastiveEmbed()
      )
      (bert): BertModelWarper(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0-11): 12 x BertLayer(
              (attention): BertAttention(
                (self): BertSdpaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (feat_map): lora.Linear(
        (base_layer): Linear(in_features=768, out_features=256, bias=True)
        (lora_dropout): ModuleDict(
          (default): Identity()
        )
        (lora_A): ModuleDict(
          (default): Linear(in_features=768, out_features=32, bias=False)
        )
        (lora_B): ModuleDict(
          (default): Linear(in_features=32, out_features=256, bias=False)
        )
        (lora_embedding_A): ParameterDict()
        (lora_embedding_B): ParameterDict()
        (lora_magnitude_vector): ModuleDict()
      )
      (input_proj): ModuleList(
        (0): Sequential(
          (0): Conv2d(192, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        )
        (1): Sequential(
          (0): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        )
        (2): Sequential(
          (0): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        )
        (3): Sequential(
          (0): Conv2d(768, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        )
      )
      (backbone): Joiner(
        (0): SwinTransformer(
          (patch_embed): PatchEmbed(
            (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
            (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          )
          (pos_drop): Dropout(p=0.0, inplace=False)
          (layers): ModuleList(
            (0): BasicLayer(
              (blocks): ModuleList(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=96, out_features=288, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=96, out_features=96, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): Identity()
                  (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=96, out_features=384, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=384, out_features=96, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=96, out_features=288, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=96, out_features=96, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.018)
                  (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=96, out_features=384, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=384, out_features=96, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
              )
              (downsample): PatchMerging(
                (reduction): Linear(in_features=384, out_features=192, bias=False)
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): BasicLayer(
              (blocks): ModuleList(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=192, out_features=576, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=192, out_features=192, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.036)
                  (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=192, out_features=768, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=768, out_features=192, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=192, out_features=576, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=192, out_features=192, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.055)
                  (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=192, out_features=768, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=768, out_features=192, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
              )
              (downsample): PatchMerging(
                (reduction): Linear(in_features=768, out_features=384, bias=False)
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              )
            )
            (2): BasicLayer(
              (blocks): ModuleList(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=384, out_features=1152, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=384, out_features=384, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.073)
                  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=384, out_features=1536, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=1536, out_features=384, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=384, out_features=1152, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=384, out_features=384, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.091)
                  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=384, out_features=1536, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=1536, out_features=384, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (2): SwinTransformerBlock(
                  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=384, out_features=1152, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=384, out_features=384, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.109)
                  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=384, out_features=1536, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=1536, out_features=384, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (3): SwinTransformerBlock(
                  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=384, out_features=1152, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=384, out_features=384, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.127)
                  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=384, out_features=1536, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=1536, out_features=384, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (4): SwinTransformerBlock(
                  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=384, out_features=1152, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=384, out_features=384, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.145)
                  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=384, out_features=1536, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=1536, out_features=384, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (5): SwinTransformerBlock(
                  (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=384, out_features=1152, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=384, out_features=384, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.164)
                  (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=384, out_features=1536, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=1536, out_features=384, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
              )
              (downsample): PatchMerging(
                (reduction): Linear(in_features=1536, out_features=768, bias=False)
                (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              )
            )
            (3): BasicLayer(
              (blocks): ModuleList(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=768, out_features=2304, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=768, out_features=768, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.182)
                  (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=768, out_features=3072, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=3072, out_features=768, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=768, out_features=2304, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=768, out_features=768, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path): DropPath(drop_prob=0.200)
                  (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=768, out_features=3072, bias=True)
                    (act): GELU(approximate='none')
                    (fc2): Linear(in_features=3072, out_features=768, bias=True)
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                )
              )
            )
          )
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (1): PositionEmbeddingSineHW()
      )
      (bbox_embed): ModuleList(
        (0-5): 6 x MLP(
          (layers): ModuleList(
            (0-1): 2 x lora.Linear(
              (base_layer): Linear(in_features=256, out_features=256, bias=True)
              (lora_dropout): ModuleDict(
                (default): Identity()
              )
              (lora_A): ModuleDict(
                (default): Linear(in_features=256, out_features=32, bias=False)
              )
              (lora_B): ModuleDict(
                (default): Linear(in_features=32, out_features=256, bias=False)
              )
              (lora_embedding_A): ParameterDict()
              (lora_embedding_B): ParameterDict()
              (lora_magnitude_vector): ModuleDict()
            )
            (2): ModulesToSaveWrapper(
              (original_module): Linear(in_features=256, out_features=4, bias=True)
              (modules_to_save): ModuleDict(
                (default): Linear(in_features=256, out_features=4, bias=True)
              )
            )
          )
        )
      )
      (class_embed): ModuleList(
        (0-5): 6 x ContrastiveEmbed()
      )
    )
  )
)
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