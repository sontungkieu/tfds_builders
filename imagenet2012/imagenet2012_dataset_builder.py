# imagenet2012_dataset_builder.py
import io, os
import tensorflow_datasets as tfds

# NEW: dùng HF streaming
from datasets import load_dataset, Image as HFImage

_DESCRIPTION = """ImageNet ILSVRC 2012 via HF streaming (no manual .tar)."""
_CITATION = """@misc{imagenet2012, title={ImageNet Large Scale Visual Recognition Challenge 2012}}"""

# Giữ phiên bản TFDS phổ biến
_VERSION = tfds.core.Version("3.0.0")

class Imagenet2012(tfds.core.GeneratorBasedBuilder):
    VERSION = _VERSION
    RELEASE_NOTES = {
        "3.0.0": "HF streaming backend; schema khớp TFDS (image, label).",
    }
    # LƯU Ý: builder gốc yêu cầu manual_dir; bản này KHÔNG cần.
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "This streaming-backed builder does not require manual downloads. "
        "Ensure you have access to ILSVRC/imagenet-1k on Hugging Face."
    )

    def _info(self):
        # TFDS official dùng ClassLabel 1000 lớp; tên lớp không bắt buộc để train
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),            # nhận bytes hoặc np.ndarray
                "label": tfds.features.ClassLabel(num_classes=1000),
                # Nếu bạn muốn có file_name như 1 số pipeline: thêm dòng sau
                # "file_name": tfds.features.Text(),
            }),
            supervised_keys=("image", "label"),
            homepage="https://image-net.org/",
            description=_DESCRIPTION,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Map split TFDS -> split HF
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
        """Stream từ HF và yield ví dụ theo schema TFDS."""
        # Yêu cầu đã accept điều khoản trên https://huggingface.co/datasets/ILSVRC/imagenet-1k
        ds = load_dataset("ILSVRC/imagenet-1k", split=hf_split, streaming=True)
        # Giữ encoded bytes thay vì PIL Image
        ds = ds.cast_column("image", HFImage(decode=False))

        for idx, ex in enumerate(ds):
            # Lấy bytes gốc (jpeg/png/webp)
            img_field = ex["image"]
            if isinstance(img_field, dict) and "bytes" in img_field:
                img_bytes = img_field["bytes"]
                fname = img_field.get("path", "")
            else:
                # Fallback: nếu lỡ decode thành PIL, re-encode JPEG
                buf = io.BytesIO()
                ex["image"].save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                fname = ""

            # Trả về theo schema TFDS: image (bytes), label (int)
            # Nếu bạn muốn lưu thêm file_name: thêm "file_name": os.path.basename(fname)
            yield idx, {
                "image": img_bytes,
                "label": int(ex["label"]),
                # "file_name": os.path.basename(fname),
            }
