import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_mask_paths(image_dir: Path, mask_dir: Path, image_ext=".bmp", mask_ext=".jpg"):
    """
    Match image files to corresponding mask files based on filename stem.
    Returns list of mask paths.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_ext)])
    mask_paths = []

    for image_file in image_files:
        base_name = Path(image_file).stem
        mask_file = base_name + mask_ext
        full_mask_path = mask_dir / mask_file
        if full_mask_path.exists():
            mask_paths.append(full_mask_path)
        else:
            print(f"⚠️ Warning: No mask found for image {image_file}")

    return mask_paths

def split_dataset(mask_paths, test_size=0.1, seed=42):
    """
    Split mask paths into train and validation sets.
    """
    return train_test_split(mask_paths, test_size=test_size, random_state=seed)

# Example usage:
if __name__ == "__main__":
    base_dir = Path("/optima/exchange/hjebril/plexilite/FAZ/split_dataset_healthy/train")
    image_dir = base_dir / "Images"
    mask_dir = base_dir / "masks"

    all_masks = load_mask_paths(image_dir, mask_dir)
    train_masks, val_masks = split_dataset(all_masks)

    print(f"Train: {len(train_masks)}, Validation: {len(val_masks)}")
