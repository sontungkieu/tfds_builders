import os
from datasets import load_dataset
import cv2
import numpy as np
from tqdm import tqdm  # Progress bar mượt
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

# Tạo dir
os.makedirs('/kaggle/working/resized_images', exist_ok=True)
output_dir = Path('/kaggle/working/resized_images')

def resize_and_save(ex):
    """Resize one image and save as PNG."""
    i, example = ex
    img_np = np.array(example['image'])
    resized = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    filename = f"{i:06d}_{example['label']}.png"
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))  # BGR for cv2.imwrite
    return {'id': i, 'image_path': str(filepath), 'label': example['label']}

# Load dataset
print("Loading HF dataset...")
dataset = load_dataset("mattymchen/celeba-hq", split='train')
total = len(dataset)

# Parallel resize với ThreadPool (multi-thread, I/O + CPU bound)
print("Starting parallel resize...")
with ThreadPoolExecutor(max_workers=8) as executor:  # Fix: 'as executor' to define variable
    futures = {executor.submit(resize_and_save, (i, ex)): i for i, ex in enumerate(dataset)}
    metadata = []
    for future in tqdm(as_completed(futures), total=total, desc="Resizing & Saving"):
        metadata.append(future.result())

# Save metadata CSV
df = pd.DataFrame(metadata)
df.to_csv('/kaggle/working/metadata.csv', index=False)
print(f"Done! {total} images resized & saved. Metadata: /kaggle/working/metadata.csv")
