import torch
import numpy as np
import cv2
import argparse
import os
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image


def generate_foreground_mask(predictor, image):

    h, w, _ = image.shape

    # Spread foreground points across center region
    input_points = np.array([
        [w//2, h//2],
        [w//2, h//3],
        [w//2, 2*h//3],
        [w//3, h//2],
        [2*w//3, h//2],
    ])

    input_labels = np.array([1, 1, 1, 1, 1])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]

    return best_mask.astype(np.uint8) * 255


def remove_foreground(image, mask):

    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask = mask / 255.0
    background_mask = 1 - mask
    background_mask = np.stack([background_mask]*3, axis=-1)

    background_only = image * background_mask

    return background_only.astype(np.uint8)


def process_folder(input_folder, mask_folder, output_folder,
                   checkpoint, model_type, device):

    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(image_files)} images")

    for filename in image_files:

        image_path = os.path.join(input_folder, filename)
        print(f"\nProcessing: {filename}")

        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print("Skipping (could not read)")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # 1️⃣ Generate mask
        foreground_mask = generate_foreground_mask(predictor, image_rgb)

        mask_save_path = os.path.join(
            mask_folder,
            os.path.splitext(filename)[0] + "_mask.png"
        )

        Image.fromarray(foreground_mask).save(mask_save_path)

        # 2️⃣ Remove foreground
        background_only = remove_foreground(image_bgr, foreground_mask)

        output_save_path = os.path.join(
            output_folder,
            os.path.splitext(filename)[0] + "_background.png"
        )

        cv2.imwrite(output_save_path, background_only)

        print("Saved mask:", mask_save_path)
        print("Saved background:", output_save_path)


def main():
    parser = argparse.ArgumentParser(description="Batch SAM Mask + Background Removal")

    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--mask_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--checkpoint", default="models/sam_vit_b.pth")
    parser.add_argument("--model_type", default="vit_b",
                        choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    process_folder(
        args.input_folder,
        args.mask_folder,
        args.output_folder,
        args.checkpoint,
        args.model_type,
        args.device
    )


if __name__ == "__main__":
    main()