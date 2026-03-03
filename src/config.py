import torch

MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

NUM_STEPS = 50
GUIDANCE_SCALE = 7.5