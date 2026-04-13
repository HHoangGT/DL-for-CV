from pathlib import Path
import shutil

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

train_images = data_dir / "train_images"
train_labels = data_dir / "train_labels"
valid_images = data_dir / "valid_images"
valid_labels = data_dir / "valid_labels"

voc_root = data_dir / "VOCdevkit" / "VOC2012"
jpeg_dir = voc_root / "JPEGImages"
mask_dir = voc_root / "SegmentationClass"
split_dir = voc_root / "ImageSets" / "Segmentation"

jpeg_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)
split_dir.mkdir(parents=True, exist_ok=True)

def copy_all(src: Path, dst: Path):
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)

copy_all(train_images, jpeg_dir)
copy_all(valid_images, jpeg_dir)
copy_all(train_labels, mask_dir)
copy_all(valid_labels, mask_dir)

train_names = sorted([p.stem for p in train_images.glob("*.jpg")])
val_names = sorted([p.stem for p in valid_images.glob("*.jpg")])

(split_dir / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
(split_dir / "val.txt").write_text("\n".join(val_names) + "\n", encoding="utf-8")

print("Done.")
print(f"JPEGImages: {len(list(jpeg_dir.glob('*.jpg')))}")
print(f"SegmentationClass: {len(list(mask_dir.glob('*.png')))}")
print(f"train.txt: {len(train_names)}")
print(f"val.txt: {len(val_names)}")