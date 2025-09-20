import os
import inspect
from functools import partial
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from attention_processor import AttnProcessor2_0,Customize_AttnProcessor2_0
from tqdm import tqdm
from diffusers import DDIMScheduler
import yaml
from datasets import DressCodeDataset
import shutil
class Config:
    def __init__(self, config_path):
        # Load YAML file
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        
        # Dynamically bind configuration data to object attributes
        for key, value in config_data.items():
            setattr(self, key, value)

# Usage example
config_path = "configs/train_dresscode_config.yaml"  # Configuration file path
config = Config(config_path)
# Function: Calculate the total number of model parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # Calculate total parameters (including trainable and non-trainable)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Calculate trainable parameters only
    return total_params, trainable_params

# Function: Convert parameter count to readable format
def readable_params(num_params):
    if num_params < 1e3:
        return f"{num_params} parameters"
    elif num_params < 1e6:
        return f"{num_params / 1e3:.2f}K parameters"
    elif num_params < 1e9:
        return f"{num_params / 1e6:.2f}M parameters"
    elif num_params < 1e12:
        return f"{num_params / 1e9:.2f}B parameters"
    else:
        return f"{num_params / 1e12:.2f}T parameters"

def compose_img_dresscode(gt_img, fake_img, im_head):
    """
    Combine real head with generated body parts based on given head image mask.
    
    Args:
    gt_img: Real image used as source for head extraction.
    fake_img: Generated image used as source for body parts.
    im_head: Binary mask of the head used to distinguish head and body.
    
    Returns:
    Composed image containing real head and generated body parts.
    """
    
    # Use im_head as binary mask for head, extract real head parts
    seg_head = im_head
    
    # Extract real head image through binary mask
    true_head = gt_img * seg_head
    
    # Select generated body parts through inverse of binary mask
    generated_body = fake_img * ~(seg_head)
    
    # Merge real head with generated body parts to form final composed image
    return true_head + generated_body 

def collate_fn(data, order):
    category = [example["category"] for example in data]
    c_name = [example["c_name"] for example in data]
    im_name = [example["im_name"] for example in data]
    image = torch.stack([example["image"] for example in data])
    texture = torch.stack([example["texture"] for example in data])
    caption = torch.cat([example["caption"] for example in data], dim=0)
    original_caption = [example["original_caption"] for example in data]
    im_sketch = torch.stack([example["im_sketch"] for example in data])
    densepose = torch.stack([example["densepose"] for example in data])
    im_mask = torch.stack([example["im_mask"] for example in data])
    inpaint_mask = torch.stack([example["inpaint_mask"] for example in data])
    result = {
        "category": category,
        "c_name": c_name,
        "im_name": im_name,
        "image": image,
        "densepose": densepose,
        "caption": caption,
        "texture": texture,
        "original_caption": original_caption,
        "im_sketch": im_sketch,
        "im_mask": im_mask,
        "inpaint_mask": inpaint_mask
    }
    if order == "test":
        stitch_label = torch.stack([example["stitch_label"] for example in data])
        result["stitch_label"] = stitch_label
    return result

def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

# Calculate linear warm-up learning rate scheduler
def lr_lambda(current_step):
    if current_step < config.warm_up_steps:
        # Linear increase of learning rate in the first 500 steps
        return float(current_step) / float(max(1, config.warm_up_steps))
    return 1.0  # Maintain initial learning rate after warm-up

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def main():

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="wandb",
        project_dir=config.output_dir
    )
    accelerator.init_trackers("my_diffusers", init_kwargs={"wandb": {"id": "3r34ekaq", "resume": "allow"}})
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")
    val_scheduler = DDIMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    # ------------------------------customize unet start------------------------------
    # Dictionary to store reconstructed attention processors
    attn_procs = {}
    for name in unet.attn_processors.keys():
        # If it's self-attention attn1, set to None, otherwise set to cross-attention dimension
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        # Record the channel size of this block at this time
        if name.startswith("mid_block"):
        # 'block_out_channels', [320, 640, 1280, 1280]
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
        # The position after up_block. in the name indicates which up block it is
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        else:
            attn_procs[name] = Customize_AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=config.ip_scale)
    # Replace attention modules: Replace unet's attention modules with reconstructed attention modules
    unet.set_attn_processor(attn_procs)
    state_dict = torch.load(config.pretrained_ip_adapter_path, map_location="cpu")
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"],strict=True)
    
    conv_new = torch.nn.Conv2d(
        # image + texture: 4, densepose: 4, im_sketch: 4, im_mask: 4, inpaint_mask: 1
        in_channels=4+4+4+4+1,
        out_channels=unet.conv_in.out_channels,
        kernel_size=3,
        padding=1,
    )
    torch.nn.init.kaiming_normal_(conv_new.weight)  
    conv_new.weight.data = conv_new.weight.data * 0.  

    conv_new.weight.data[:, :9] = unet.conv_in.weight.data  
    conv_new.bias.data = unet.conv_in.bias.data  

    unet.conv_in = conv_new  # replace conv layer in unet
    unet.config['in_channels'] = 17  # update config
    unet.config.in_channels = 17  # update config

    # Freeze most parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Unfreeze conv_in, self-attention layers and IP-adapter
    unet.conv_in.requires_grad_(True)
    for name, param in unet.named_modules():
        if "attn1" in name or "attn2.processor" in name:
            param.requires_grad_(True)

    # Calculate total and trainable parameter counts
    total_params, trainable_params = count_parameters(unet)
    # Print results
    accelerator.print(f"UNet total parameters: {readable_params(total_params)}")
    accelerator.print(f"UNet trainable parameters: {readable_params(trainable_params)}")
    # ------------------------------customize unet end------------------------------

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # For mixed precision training, we convert text_encoder and vae weights to half precision
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    val_scheduler.set_timesteps(num_inference_steps=config.num_inference_steps,device=accelerator.device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # dataloader
    train_dataset = DressCodeDataset(
        dataroot_path=config.dataroot_path,
        phase='train',
        order="paired",
        sketch_threshold_range=(20, 20),
        radius=5,
        tokenizer=tokenizer,
        size=(512, 384),
        uncond_prob=config.uncond_prob,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.dataloader_num_workers, 
        collate_fn=partial(collate_fn, order="train"),
    )
    # Prepare everything with our `accelerator`.
    unet, optimizer, scheduler, train_dataloader = accelerator.prepare(unet, optimizer, scheduler, train_dataloader)
    global_step = 0
    if config.resume_state != "":
        global_step = int(config.resume_state.split("/")[-1].split("-")[1])
        accelerator.load_state(config.resume_state)
        accelerator.print("Resume from state:", config.resume_state)
    
    unet.train()
    accelerator.print("Start training from global_step:", global_step)
    # Create progress bar with tqdm
    with tqdm(total=config.max_train_steps - global_step, desc="Step", unit="global_step", colour='green') as pbar:
        while global_step < config.max_train_steps:
            epoch_loss = 0.0
            for _, batch in enumerate(train_dataloader):
                with accelerator.autocast():
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["caption"])[0]
                        # During training, all except inpaint_mask need to concatenate texture
                        image_latent = vae.encode(batch["image"]).latent_dist.sample()
                        image_latent = image_latent * vae.config.scaling_factor
                        
                        texture_latent = vae.encode(batch["texture"]).latent_dist.sample()
                        texture_latent = texture_latent * vae.config.scaling_factor

                        model_input = torch.cat([image_latent, torch.zeros_like(image_latent)], dim = config.concat_dim)
                    

                        densepose_latent = vae.encode(batch["densepose"]).latent_dist.sample()
                        densepose_latent = densepose_latent * vae.config.scaling_factor
                        densepose_latent_input = torch.cat([densepose_latent, texture_latent], dim=config.concat_dim)

                        im_mask_latent = vae.encode(batch["im_mask"]).latent_dist.sample()
                        im_mask_latent = im_mask_latent * vae.config.scaling_factor
                        im_mask_latent_input = torch.cat([im_mask_latent, texture_latent], dim = config.concat_dim)

                        inpaint_mask = torch.stack(
                            [
                                torch.nn.functional.interpolate(batch["inpaint_mask"],size=tuple(x // 8 for x in config.size))
                            ]
                        )
                        # Masks not on the human body do not concatenate texture
                        inpaint_mask = inpaint_mask.reshape(-1, 1, config.size[0] // 8, config.size[1] // 8)
                        inpaint_mask_input = torch.cat([inpaint_mask, torch.zeros_like(inpaint_mask)], dim=config.concat_dim)

                        im_sketch_latent = vae.encode(batch["im_sketch"]).latent_dist.sample()
                        im_sketch_latent = im_sketch_latent * vae.config.scaling_factor
                        im_sketch_input = torch.cat([im_sketch_latent, texture_latent], dim = config.concat_dim)

            
                        noise = torch.randn_like(model_input)
                        bsz = model_input.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                        # Only add noise to real images
                        noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)
                        latent_model_input = torch.cat(
                            [
                                noisy_latents, 
                                im_mask_latent_input,
                                inpaint_mask_input,
                                im_sketch_input,
                                densepose_latent_input
                            ], 
                            dim = 1
                        )

                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    if accelerator.num_processes > 1:
                        # Gather the losses across all processes for logging (if we use distributed training).
                        avg_loss = accelerator.gather(loss).sum().item()   
                    else:
                        avg_loss = loss.item()
                    epoch_loss += avg_loss

                    accelerator.backward(loss)
                    optimizer.step()
                    # Update learning rate
                    scheduler.step()
                    optimizer.zero_grad()  
                    # ---------------------------------------------------------------------------------------
                    if (global_step + 1) % config.log_info_every_n_steps == 0:
                        if accelerator.is_main_process:
                            accelerator.log({"global_step_loss": avg_loss})
                            accelerator.log({"lr": optimizer.param_groups[0]["lr"]})
                    
                    if (global_step + 1) == config.max_train_steps:
                        if accelerator.is_main_process:
                            save_path = os.path.join(config.output_dir, f"dresscode_checkpoint-{global_step}-step")
                            accelerator.save_state(save_path)
                        accelerator.print("Training finished")
                        accelerator.end_training()
                        return
                global_step += 1
                # Update progress bar
                pbar.update(1)
            if accelerator.is_main_process:
                # New: Save latest model (regardless of whether it's best)
                # Delete old latest model
                for dir_name in os.listdir(config.output_dir):
                    dir_path = os.path.join(config.output_dir, dir_name)
                    if dir_name.startswith("latest") and os.path.isdir(dir_path):
                        accelerator.print(f"🗑️ Deleting old latest model directory: {dir_path}")
                        shutil.rmtree(dir_path)
                    
                # Save new latest model
                latest_save_path = os.path.join(
                    config.output_dir, 
                    f"latest_checkpoint-{global_step}-step"
                )
                accelerator.save_state(latest_save_path)
                accelerator.print(f"📌 Saved latest model to: {latest_save_path}")

if __name__ == "__main__":
    main()