CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/train_unianimate_wan.py \
   --task train  \
   --train_architecture lora \
   --lora_rank 64 --lora_alpha 64  \
   --dataset_path data/example_dataset   \
   --output_path ./models_out_one_GPU   \
   --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"    \
   --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    \
   --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    \
   --max_epochs 10   --learning_rate 1e-4   \
   --accumulate_grad_batches 1   \
   --use_gradient_checkpointing --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload
   