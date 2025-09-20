import os
import inspect
import torch
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from attention_processor import AttnProcessor2_0,Customize_AttnProcessor2_0
from tqdm import tqdm
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import yaml
from datasets import VitonHDDataset
class Config:
    def __init__(self, config_path):
        # Load YAML file
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        
        # Dynamically bind configuration data to object attributes
        for key, value in config_data.items():
            setattr(self, key, value)

# Usage example
config_path = "configs/inference_vitonhd_config.yaml"  # Configuration file path
config = Config(config_path)


def collate_fn(data):
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
    im_parse = torch.stack([example["im_parse"] for example in data])
    result = {
        "c_name": c_name,
        "im_name": im_name,
        "image": image,
        "texture": texture,
        "caption": caption,
        "original_caption": original_caption,
        "im_sketch": im_sketch,
        "densepose": densepose,
        "im_mask": im_mask,
        "inpaint_mask": inpaint_mask,
        "im_parse": im_parse
    }
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

def compose_img(gt_img, fake_img, im_parse):

    seg_head = torch.logical_or(im_parse == 1, im_parse == 2)
    seg_head = torch.logical_or(seg_head, im_parse == 4)
    seg_head = torch.logical_or(seg_head, im_parse == 13)

    true_head = gt_img * seg_head
    true_parts = true_head

    generated_body = (transforms.functional.pil_to_tensor(fake_img).cuda() / 255) * (~(seg_head))

    return true_parts + generated_body

def main():

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
    )

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
            
    # Set unet's attention processor to the reconstructed attention dictionary
    unet.set_attn_processor(attn_procs)
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
    
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
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
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
    extra_step_kwargs = prepare_extra_step_kwargs(val_scheduler, generator, eta=0.0)
    
    test_dataset = VitonHDDataset(
        dataroot_path=config.dataroot_path,
        phase='test',
        order=config.order,
        sketch_threshold_range=(20, 20),
        radius=5,
        tokenizer=tokenizer,
        size=(512, 384),
        uncond_prob=config.uncond_prob,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size, 
        shuffle=False, 
        num_workers=config.dataloader_num_workers, 
        collate_fn=collate_fn,
    )
    # Prepare everything with our `accelerator`.
    unet, test_dataloader = accelerator.prepare(unet, test_dataloader)
    accelerator.load_state(config.resume_state)
    config.inference_grid = config.inference_grid.format(config.order)
    config.inference_one = config.inference_one.format(config.order)
    os.makedirs(os.path.join(config.output_dir, config.inference_grid), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, config.inference_one), exist_ok=True)
    # Inference mode
    unet.eval()
    with torch.no_grad():
        with accelerator.autocast():
            pbar1 = tqdm(total=len(test_dataloader), position=0, desc="Inference", leave=False, colour='green', ncols=100, unit="batch")
            for idx, batch in enumerate(test_dataloader):
                # 1. Text
                bs = batch["caption"].shape[0]
                encoder_hidden_states = text_encoder(batch["caption"])[0]
                uncond_encoder_hidden_states = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt").input_ids
                uncond_encoder_hidden_states = text_encoder(uncond_encoder_hidden_states.to(device=accelerator.device))[0]
                uncond_encoder_hidden_states = torch.cat([uncond_encoder_hidden_states] * bs)

                text_embeddings = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states])
                # 2. Texture
                texture_latent = vae.encode(batch["texture"]).latent_dist.sample()
                texture_latent = texture_latent * vae.config.scaling_factor
                # 3. Pose
                densepose_latent = vae.encode(batch["densepose"]).latent_dist.sample()
                densepose_latent = densepose_latent * vae.config.scaling_factor
                densepose_latent_input = torch.cat([densepose_latent, texture_latent], dim=config.concat_dim)
                uncond_densepose_latent_input = torch.zeros_like(densepose_latent_input)
                densepose_latent_input = torch.cat([uncond_densepose_latent_input, densepose_latent_input])
                # 4. Masked model
                im_mask_latent = vae.encode(batch["im_mask"]).latent_dist.sample()
                im_mask_latent = im_mask_latent * vae.config.scaling_factor
                im_mask_latent_input = torch.cat([im_mask_latent, texture_latent], dim = config.concat_dim)

                uncond_im_mask_latent_input = torch.cat([im_mask_latent, torch.zeros_like(im_mask_latent)], dim = config.concat_dim)
                im_mask_latent_input = torch.cat([uncond_im_mask_latent_input, im_mask_latent_input])

                # 5. Inpaint mask
                inpaint_mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(batch["inpaint_mask"],size=tuple(x // 8 for x in config.size))
                    ]
                )
                inpaint_mask = inpaint_mask.reshape(-1, 1, config.size[0] // 8, config.size[1] // 8)

                inpaint_mask_input = torch.cat([inpaint_mask, torch.zeros_like(inpaint_mask)], dim=config.concat_dim)
                inpaint_mask_input = torch.cat([inpaint_mask_input] * 2)

                # 6. Sketch
                im_sketch_latent = vae.encode(batch["im_sketch"]).latent_dist.sample()
                im_sketch_latent = im_sketch_latent * vae.config.scaling_factor
                im_sketch_input = torch.cat([im_sketch_latent, texture_latent], dim = config.concat_dim)
                uncond_im_sketch_input = torch.zeros_like(im_sketch_input)
                im_sketch_input = torch.cat([uncond_im_sketch_input, im_sketch_input])

                latents = randn_tensor(
                    uncond_im_mask_latent_input.shape,
                    generator=generator,
                    device=uncond_im_mask_latent_input.device,
                    dtype=weight_dtype,
                )
                
                timesteps = val_scheduler.timesteps
                latents = latents * val_scheduler.init_noise_sigma
                num_warmup_steps = (len(timesteps) - config.num_inference_steps * val_scheduler.order)
                with tqdm(total=config.num_inference_steps) as progress_bar:
                    # Loop through the sampling timesteps
                    for i, t in tqdm(enumerate(timesteps)):
                        # Prepare model input
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = val_scheduler.scale_model_input(latent_model_input, t)
                        inpainting_latent_model_input = torch.cat(
                            [
                                latent_model_input, 
                                im_mask_latent_input,
                                inpaint_mask_input,
                                im_sketch_input,
                                densepose_latent_input
                            ], 
                            dim = 1
                        )
                        noise_pred = unet(inpainting_latent_model_input, t, text_embeddings).sample
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        # compute the previous noisy sample x_t -> x_t-1
                        latents = val_scheduler.step(
                            noise_pred, t, latents, **extra_step_kwargs
                        ).prev_sample
                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % val_scheduler.order == 0):
                            progress_bar.update()
                # Decode the final latents
                latents = latents.split(latents.shape[config.concat_dim] // 2, dim=config.concat_dim)[0]
                latents = 1 / vae.config.scaling_factor * latents
                image = vae.decode(latents.to(accelerator.device, dtype=weight_dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = numpy_to_pil(image)
                for i in range(len(image)):
                    model_img = batch["image"][i] * 0.5 + 0.5
                    final_img = compose_img(model_img, image[i], batch["im_parse"][i])
                    final_img1 = transforms.functional.to_pil_image(final_img)
                    final_img1.save(os.path.join(config.output_dir,config.inference_one,batch["im_name"][i]))
                    all_img = []
                    # Map data from [-1, 1] range to [0, 1]
                    def normalize_to_01(img):
                        return (img + 1) / 2

                    all_img.append(normalize_to_01(batch["image"][i].unsqueeze(0)))
                    all_img.append(final_img.unsqueeze(0))  # final_img is already [0, 1], no processing needed
                    all_img.append(normalize_to_01(batch["texture"][i].unsqueeze(0)))
                    all_img.append(normalize_to_01(batch["im_sketch"][i].unsqueeze(0)))
                    all_img.append(normalize_to_01(batch["densepose"][i].unsqueeze(0)))
                    all_img.append(normalize_to_01(batch["im_mask"][i].unsqueeze(0)))

                    # Generate grid image
                    grid_image = make_grid(torch.cat(all_img, dim=0), nrow=len(all_img), padding=10, normalize=True)
                    to_pil_image(grid_image).save(os.path.join(config.output_dir,config.inference_grid, batch["im_name"][i]))

                torch.cuda.empty_cache()
                pbar1.update(1)

if __name__ == "__main__":


    main()