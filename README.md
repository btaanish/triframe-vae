# Hybrid SDF-Triplane VAE with EDM for 3D Shape Generation

A **PyTorch implementation** of a two-stage text-to-3D generation combining:

- **Stage 1:** Hybrid SDF-Triplane VAE for 3D shape encoding  
- **Stage 2:** EDM latent diffusion for text conditioned generation  

This implementation builds upon the initial **VQ-VAE** with **SDFusion**, integrating continuous latent spaces, triplane representation, and EDM training.

---

## Training the Triplane VAE

```bash
python train_stage1_triplane_vae.py \
    --name triplane_vae_softnet \
    --dataset_mode softnet_sdf \
    --dataroot "data/Softnet/SDF_v1/resolution_128" \
    --cat all \
    --res 128 \
    --batch_size 2 \
    --lr 0.0001 \
    --total_iters 200000 \
    --save_steps_freq 10000 \
    --print_freq 100 \
    --display_freq 1000 \
    --gpu_ids 0
````

---

## Training the EDM Diffusion

```bash
python train_stage2_edm_diffusion.py \
    --name edm_diffusion_softnet \
    --dataset_mode softnet_sdf \
    --dataroot "data/Softnet/SDF_v1/resolution_128" \
    --cat all \
    --res 128 \
    --vq_ckpt checkpoints/triplane_vae_softnet/triplane_vae_epoch-best.pth \
    --batch_size 2 \
    --lr 0.0001 \
    --total_iters 300000 \
    --save_steps_freq 10000 \
    --print_freq 100 \
    --display_freq 1000 \
    --gpu_ids 0
```

---

##  Sample Inference

```bash
python inference_txt2shape.py
```
