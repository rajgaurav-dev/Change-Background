import torch
from diffusers import StableDiffusionInpaintPipeline
from config import MODEL_ID, DEVICE, DTYPE

def load_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE
    )

    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

    return pipe