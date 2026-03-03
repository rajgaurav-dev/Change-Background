import os
import argparse
from PIL import Image
from tqdm import tqdm

from pipeline import load_pipeline
from config import NUM_STEPS, GUIDANCE_SCALE
from prompts import BACKGROUND_PROMPTS


def generate_backgrounds(input_image, mask_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline()

    image = Image.open(input_image).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    print("Generating backgrounds...")

    for idx, prompt in enumerate(tqdm(BACKGROUND_PROMPTS)):

        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE
        ).images[0]

        output_path = os.path.join(output_dir, f"generated_{idx}.png")
        result.save(output_path)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting Generator")

    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    generate_backgrounds(
        input_image=args.image,
        mask_path=args.mask,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()