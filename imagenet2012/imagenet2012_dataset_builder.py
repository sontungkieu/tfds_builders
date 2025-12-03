# imagenet2012_dataset_builder.py
# -*- coding: utf-8 -*-
"""
ImageNet ILSVRC 2012 via Hugging Face streaming, for TFDS build in kvfrans/tfds_builders.

- Không cần tải .tar thủ công; yêu cầu bạn đã accept điều khoản trên
  https://huggingface.co/datasets/ILSVRC/imagenet-1k và đã login bằng token HF.
- Giữ schema TFDS: {"image": Image(), "label": ClassLabel(num_classes=1000)}
- Tránh import 'datasets' ở mức module để không bị shadow bởi thư mục/tập tin cục bộ tên 'datasets'.

Cách dùng:
  pip install -U "datasets>=2.18,<3.0" "huggingface_hub>=0.34,<2.0"
  python -c "from huggingface_hub import login; login()"
  tfds build  # trong thư mục chứa file này
"""

import io
import os
import tensorflow_datasets as tfds


_DESCRIPTION = """ImageNet ILSVRC 2012 streamed from Hugging Face (no manual .tar)."""
_CITATION = r"""
@misc{imagenet2012,
  title  = {ImageNet Large Scale Visual Recognition Challenge 2012},
  author = {Russakovsky et al.},
  year   = {2015}
}
"""


class Builder(tfds.core.GeneratorBasedBuilder):
    """TFDS community builder for ImageNet2012 using HF streaming."""
    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "3.0.0": "Initial streaming-backed release (schema matches TFDS: image, label).",
    }

    # Không cần manual_dir nữa
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "No manual downloads. Ensure you have access to ILSVRC/imagenet-1k on Hugging Face "
        "and you are logged in (use `from huggingface_hub import login; login()`)."
    )

    def _info(self) -> tfds.core.DatasetInfo:
        # Giữ schema giống TFDS official: image bytes + int label (1000 lớp)
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),                     # encoded bytes -> decoded by TFDS
                "label": tfds.features.ClassLabel(num_classes=1000),
                # Nếu pipeline của bạn cần tên file gốc:
                # "file_name": tfds.features.Text(),
            }),
            supervised_keys=("image", "label"),
            homepage="https://image-net.org/",
            description=_DESCRIPTION,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Map TFDS split -> HF split
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"hf_split": "train"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"hf_split": "validation"},  # HF dùng 'validation'
            ),
        ]

    def _generate_examples(self, hf_split):
        """Yields (key, example) from HF streaming (no local tar needed)."""
        # Import ở đây để tránh shadow 'datasets' cục bộ
        import importlib
        hfd = importlib.import_module("datasets")  # Hugging Face 'datasets'

        # Giữ bytes gốc (không decode về PIL ở Python)
        ds = hfd.load_dataset("ILSVRC/imagenet-1k", split=hf_split, streaming=True)
        HFImage = hfd.Image
        ds = ds.cast_column("image", HFImage(decode=False))

        for idx, ex in enumerate(ds):
            img_field = ex["image"]
            if isinstance(img_field, dict) and "bytes" in img_field:
                img_bytes = img_field["bytes"]
                fname = img_field.get("path", "")
            else:
                # Fallback nếu lỡ decode: re-encode JPEG
                buf = io.BytesIO()
                ex["image"].save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                fname = ""

            # Khóa record (key) ưu tiên dùng basename để ổn định; nếu thiếu thì synthesize
            key = os.path.basename(fname) if fname else f"{hf_split}_{idx:08d}.JPEG"

            # Trả về đúng schema TFDS
            example = {
                "image": img_bytes,
                "label": int(ex["label"]),
                # "file_name": os.path.basename(fname),
            }
            yield key, example


# Để TFDS CLI dễ tìm class
__all__ = ["Builder"]
