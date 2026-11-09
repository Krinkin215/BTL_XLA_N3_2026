"""Prepare DSB2018 instance labels without modifying the source dataset.

Dependencies: numpy, Pillow. Run --help for local/Colab usage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 42
PROFILE = "grayscale_dark_background"


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path):
    with path.open("rb") as f, Image.open(f) as im:
        mode = im.mode
        arr = np.asarray(im).copy()
        rgb = np.asarray(im.convert("RGB")).copy()
    return arr, rgb, mode


def png_files(folder):
    return sorted((p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith('.png')),
                  key=lambda p: p.name)


def merge_masks(sample, shape):
    files = png_files(sample / "masks")
    if not files or len(files) > 65535:
        raise ValueError(f"Invalid mask count: {len(files)}")
    labels = np.zeros(shape, dtype=np.uint16)
    mapping = []
    for label_id, path in enumerate(files, 1):
        with path.open("rb") as f, Image.open(f) as im:
            a = np.asarray(im).copy()
        if a.ndim != 2 or a.shape != shape:
            raise ValueError(f"Mask is not a matching 2D array: {path.name}")
        if not set(np.unique(a)).issubset({0, 255}):
            raise ValueError(f"Mask has unexpected values: {path.name}")
        binary = a > 0
        area = int(binary.sum())
        if area == 0:
            raise ValueError(f"Empty mask: {path.name}")
        overlap = int(np.count_nonzero(binary & (labels > 0)))
        if overlap:
            raise ValueError(f"Overlapping mask: {path.name}; pixels={overlap}")
        labels[binary] = label_id
        mapping.append(dict(source_id=sample.name, instance_id=label_id,
                            source_mask=path.name, area_pixels=area))
    return labels, mapping


def normalize_gray(gray, low=1, high=99.8):
    a = gray.astype(np.float32)
    p1, p998 = np.percentile(a, [low, high])
    if p998 <= p1:
        raise ValueError("Percentile normalization has zero range")
    norm = np.clip((a - p1) / (p998 - p1), 0, 1).astype(np.float32)
    return norm, float(p1), float(p998)


def appearance(rgb):
    # Exact channel equality is a reproducible appearance rule, not a stain classifier.
    achromatic = bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and
                      np.array_equal(rgb[..., 0], rgb[..., 2]))
    gray = rgb[..., 0] if achromatic else np.asarray(Image.fromarray(rgb).convert("L"))
    h, w = gray.shape
    bh, bw = max(1, h // 20), max(1, w // 20)
    border = np.concatenate([gray[:bh].ravel(), gray[-bh:].ravel(),
                             gray[:, :bw].ravel(), gray[:, -bw:].ravel()])
    border_median = float(np.median(border))
    group = ("grayscale_dark_background" if border_median < 127.5 else
             "grayscale_light_background") if achromatic else "color"
    return group, gray, border_median


def split_records(records):
    # Pixel-identical RGB images remain in the same split, even if source IDs differ.
    groups = defaultdict(list)
    for r in records:
        groups[r["pixel_sha256"]].append(r)
    strata = defaultdict(list)
    for digest, members in sorted(groups.items()):
        r = members[0]
        density_bin = "few" if r["num_instances"] < 20 else (
            "medium" if r["num_instances"] < 60 else "many")
        strata[(r["appearance_group"], density_bin)].append((digest, members))
    rng = random.Random(SEED)
    for key, buckets in sorted(strata.items()):
        rng.shuffle(buckets)
        # Allocate whole duplicate groups; rounding means global ratios are approximate.
        n = len(buckets)
        a, b = int(n * .6), int(n * .2)
        for i, (_, members) in enumerate(buckets):
            split = "train" if i < a else "val" if i < a + b else "test"
            for r in members:
                r["split"] = split
    # A compact, deterministic starting experiment. Selection uses appearance only;
    # density stratification was already used above. No segmentation score is used.
    for split, target in [("train", 60), ("val", 20), ("test", 20)]:
        eligible = sorted((r for r in records if r["split"] == split and
                           r["appearance_group"] == PROFILE), key=lambda x: x["source_id"])
        rng.shuffle(eligible)
        for r in eligible[:target]:
            r["primary_100"] = 1


def label_rgb(labels):
    rng = np.random.default_rng(SEED)
    colors = rng.integers(50, 256, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[labels]


def boundary(labels):
    edges = np.zeros(labels.shape, dtype=bool)
    edges[1:] |= labels[1:] != labels[:-1]
    edges[:-1] |= labels[:-1] != labels[1:]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edges & (labels > 0)


def make_previews(out, records):
    results = out / "results"
    results.mkdir()
    panels = []
    for group in sorted({r["appearance_group"] for r in records}):
        candidates = sorted((r for r in records if r["appearance_group"] == group),
                            key=lambda r: r["num_instances"])
        for index in sorted({0, len(candidates) // 2, len(candidates) - 1}):
            panels.append(candidates[index])
    tile, row_h = 280, 320
    canvas = Image.new("RGB", (tile * 3, row_h * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    preview_ids = []
    for j, r in enumerate(panels):
        _, rgb, _ = load_image(out / r["image_path"])
        labels = np.asarray(Image.open(out / r["mask_path"]))
        overlay = rgb.copy()
        overlay[boundary(labels)] = [255, 50, 40]
        for k, (arr, title) in enumerate([(rgb, "Original"), (label_rgb(labels), "Instance GT"),
                                          (overlay, "GT boundary overlay")]):
            im = Image.fromarray(arr)
            im.thumbnail((tile - 12, tile - 12), Image.Resampling.NEAREST if k == 1 else Image.Resampling.LANCZOS)
            x, y = k * tile, j * row_h
            canvas.paste(im, (x + (tile - im.width) // 2, y + 22))
            draw.text((x + 6, y + 3), title, fill="black")
        draw.text((6, (j + 1) * row_h - 30),
                  f'{r["source_id"][:16]} | {r["appearance_group"]} | n={r["num_instances"]}', fill="black")
        preview_ids.append(dict(source_id=r["source_id"], appearance_group=r["appearance_group"],
                                num_instances=r["num_instances"], split=r["split"]))
    canvas.save(results / "quality_control.png")
    write_csv(results / "preview_samples.csv", preview_ids)
    r = next(r for r in records if r["primary_100"])
    _, rgb, _ = load_image(out / r["image_path"])
    gray = rgb[..., 0]
    norm, p1, p998 = normalize_gray(gray)
    compare = Image.new("RGB", (840, 450), "white")
    d = ImageDraw.Draw(compare)
    for x, a, title in [(0, gray, "Original grayscale (display range 0..255)"),
                         (420, (norm * 255).astype(np.uint8), "Normalized P1..P99.8 (display range 0..1)")]:
        im = Image.fromarray(a).convert("RGB")
        im.thumbnail((400, 380))
        compare.paste(im, (x + 10, 35))
        d.text((x + 10, 10), title, fill="black")
    d.text((10, 425), f'{r["source_id"][:20]} | P1={p1:.3f}, P99.8={p998:.3f}', fill="black")
    compare.save(results / "normalization_example.png")


def verify(out, records):
    counts = Counter()
    hash_splits = defaultdict(set)
    for r in records:
        image, rgb, _ = load_image(out / r["image_path"])
        with Image.open(out / r["mask_path"]) as im:
            labels = np.asarray(im)
        if labels.dtype != np.uint16 or labels.shape != image.shape[:2]:
            raise ValueError(f'Invalid output TIFF: {r["source_id"]}')
        if not np.array_equal(np.unique(labels), np.arange(r["num_instances"] + 1)):
            raise ValueError(f'Missing instance IDs: {r["source_id"]}')
        if sha256(out / r["image_path"]) != r["source_file_sha256"]:
            raise ValueError("Copied image differs from source")
        if r["normalized_path"]:
            norm = np.load(out / r["normalized_path"])
            if norm.dtype != np.float32 or norm.shape != labels.shape or not np.isfinite(norm).all():
                raise ValueError("Invalid normalized array")
            if norm.min() < 0 or norm.max() > 1:
                raise ValueError("Normalization out of range")
        hash_splits[r["pixel_sha256"]].add(r["split"])
        counts[r["split"]] += 1
    if any(len(s) != 1 for s in hash_splits.values()):
        raise ValueError("Exact duplicate image crosses splits")
    for split, n in counts.items():
        for folder, suffix in [("images", ".png"), ("masks", ".tif")]:
            if len(list((out / "processed_data" / split / folder).glob(f"*{suffix}"))) != n:
                raise ValueError("Unexpected or missing files")
    return dict(status="PASS", checked_pairs=len(records),
                image_copy_sha256="PASS", uint16_tiff_roundtrip="PASS",
                source_mask_reconstruction="PASS", normalized_float32_range="PASS",
                exact_duplicate_split_leakage="NONE", split_counts=dict(counts))


def summary_rows(records):
    rows = []
    for split in ["train", "val", "test"]:
        subset = [r for r in records if r["split"] == split]
        rows.append(dict(split=split, num_images=len(subset),
                         num_instances=sum(r["num_instances"] for r in subset),
                         primary_100=sum(r["primary_100"] for r in subset)))
    return rows


def pretrained_profile(out, records):
    """Keep the existing validation/test assignments for fixed-weight inference."""
    selected = [r for r in records if r['primary_100'] and r['split'] in ('val', 'test')]
    folder = out / 'experiments/pretrained_fluo'
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / 'manifest.csv', selected)
    for split in ['val', 'test']:
        ids = sorted(r['source_id'] for r in selected if r['split'] == split)
        (folder / f'{split}_ids.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
    config = dict(mode='pretrained_fixed_weights', proposed_model='2D_versatile_fluo',
                  model_selection_basis='Single-channel fluorescent-nuclei model; visual modality review still required',
                  manifest='experiments/pretrained_fluo/manifest.csv',
                  validation_images=sum(r['split'] == 'val' for r in selected),
                  test_images=sum(r['split'] == 'test' for r in selected),
                  training=False, fine_tuning=False, seed=SEED,
                  input='normalized_path: float32 HxW, clipped P1/P99.8 per image; do not normalize twice',
                  thresholds='Start with model defaults; any changes must be selected on validation and frozen before test',
                  independent_generalization_evaluation=False,
                  reason='Official StarDist documentation states both 2D_versatile_fluo and 2D_paper_dsb2018 were trained on a subset of DSB2018',
                  official_source='https://github.com/stardist/stardist#pretrained-models-for-2d')
    (folder / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    protocol = f"""# Hướng dẫn sử dụng StarDist pretrained

Thí nghiệm sử dụng mô hình pretrained với trọng số có sẵn, không huấn luyện hoặc fine-tune. Mô hình đề xuất là **2D_versatile_fluo**, dành cho ảnh nhân huỳnh quang đơn kênh theo tài liệu StarDist [1]. Cần kiểm tra trực quan để xác nhận miền ảnh; nhóm xám nền tối được phân loại theo diện mạo pixel. Định dạng RGB không đủ để xác định ảnh H&E.

## Tập dữ liệu thực nghiệm

- Manifest: `experiments/pretrained_fluo/manifest.csv`.
- Validation: **{config['validation_images']} ảnh** để kiểm tra đầu vào, chọn tiền xử lý/ngưỡng StarDist, Otsu và Watershed.
- Test: **{config['test_images']} ảnh** để so sánh cuối cùng sau khi chốt cấu hình trên validation.
- Không cần tập train để chạy trọng số có sẵn. 60 ảnh train của tập grayscale100 và phần train đầy đủ vẫn được giữ dự phòng, chưa được sử dụng trong protocol này.
- Giữ nguyên ID và split; không chuyển ảnh train sang test hay chia lại theo kết quả dự đoán. Các thuật toán sử dụng cùng danh sách ID ở thư mục trên.

Tập thực nghiệm ban đầu gồm 40 ảnh. Khi mở rộng, lấy thêm ảnh từ cùng split gốc và chốt danh sách trước khi xem điểm test.

## Đầu vào và cách tải mô hình

`normalized_path` trỏ tới NPY float32 H×W đã chuẩn hóa P1/P99.8 và clip [0,1]. Mỗi ảnh dùng percentile riêng; không chuẩn hóa thêm lần nữa. Ảnh gốc và TIFF ground truth vẫn có trong manifest. Pipeline chuẩn hóa này được chọn minh bạch cho thí nghiệm; không mặc định giống mọi ví dụ StarDist.

Ví dụ chạy dự đoán trên Colab (mô hình chưa được thực thi trong gói dữ liệu này):

```python
# Ô Colab riêng: %pip install tensorflow stardist
from pathlib import Path
import pandas as pd
import numpy as np
from stardist.models import StarDist2D

project = Path('/content/StarDist_Dataset')
df = pd.read_csv(project / 'experiments/pretrained_fluo/manifest.csv')
row = df[df['split'] == 'val'].iloc[0]  # bắt đầu trên validation
image = np.load(project / row['normalized_path'])
model = StarDist2D.from_pretrained('2D_versatile_fluo')
predicted_instances, details = model.predict_instances(image)
print(predicted_instances.shape, predicted_instances.max())
```

Khởi đầu với ngưỡng đi kèm mô hình. Nếu thay đổi ngưỡng xác suất/NMS, chọn trên validation và lưu giá trị trước khi chạy test. Khi so sánh instance, sử dụng cùng ngưỡng IoU matching, chẳng hạn 0.5, cùng precision/recall/F1 và TP/FP/FN; không gọi trung bình IoU pixel hoặc accuracy của matching là COCO AP. Gói dữ liệu chưa chứa kết quả so sánh các thuật toán.

## Giới hạn đánh giá

Theo tài liệu chính thức, **cả 2D_versatile_fluo và 2D_paper_dsb2018 đều đã được huấn luyện trên một phần DSB2018** [1]. Bản chia val/test cục bộ tách dữ liệu chỉnh tham số khỏi dữ liệu đánh giá cuối cùng; chưa xác định ảnh test có xuất hiện trong dữ liệu huấn luyện mô hình pretrained hay không.

Thí nghiệm sử dụng StarDist pretrained với trọng số cố định trên tập ảnh DSB2018 được chọn. Do chưa đối chiếu chính xác danh sách huấn luyện gốc, kết quả chỉ phục vụ minh họa và so sánh trong phạm vi dữ liệu này, chưa phải ước lượng khả năng tổng quát hóa độc lập.

[1] https://github.com/stardist/stardist#pretrained-models-for-2d
"""
    (out / 'HUONG_DAN_PRETRAINED.md').write_text(protocol, encoding='utf-8')
    return config


def notebook(out):
    def md(s):
        return dict(cell_type="markdown", metadata={}, source=s.splitlines(keepends=True))
    def code(s):
        return dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=s.splitlines(keepends=True))
    cells = [md("# Chuẩn bị dữ liệu StarDist\nNotebook dùng trên máy cá nhân hoặc Google Colab. Gói dữ liệu đã được xử lý và kiểm tra; có thể sử dụng trực tiếp hoặc tái tạo từ dữ liệu gốc.\n\n**Colab:** giải nén gói ZIP vào `/content`, mở notebook và đặt `PROJECT` trỏ tới thư mục chứa `dataset_manifest.csv`. Quy trình chuẩn bị dữ liệu không cần GPU."),
             code("%pip -q install numpy Pillow pandas\n"),
             code("from pathlib import Path\nimport json, sys, subprocess\nimport numpy as np\nimport pandas as pd\nfrom PIL import Image\nfrom IPython.display import display\nPROJECT = Path.cwd()  # Colab: Path('/content/StarDist_Dataset')\nassert (PROJECT / 'dataset_manifest.csv').exists(), 'Sửa PROJECT thành thư mục gói dữ liệu'\n"),
             md("## Tái tạo từ dữ liệu gốc (tùy chọn)\nMỗi PNG mask tương ứng một nhân. Script kiểm tra mask rỗng, sai kích thước, sai giá trị và chồng lấn; sau đó ghi TIFF uint16, nền 0 và ID 1..N. Chia theo nhóm ảnh trùng pixel, phân tầng theo diện mạo và số nhân, seed 42. Giữ nguyên ảnh PNG, không resize.\n\nĐể chạy lại: tải `stage1_train.zip` lên Colab và giải nén; sửa `RAW`, `NEW_OUTPUT`, bật `RUN_PREPARATION`. Đầu ra phải là thư mục mới để tránh trộn kết quả."),
             code("RUN_PREPARATION = False\nRAW = Path('/content/stage1_train')\nNEW_OUTPUT = Path('/content/StarDist_Project_rebuilt')\nif RUN_PREPARATION:\n    subprocess.run([sys.executable, str(PROJECT / 'prepare_dataset.py'), '--raw', str(RAW), '--out', str(NEW_OUTPUT)], check=True)\n"),
             md("## Thống kê và danh sách dữ liệu pretrained\nThí nghiệm sử dụng pretrained, không train/fine-tune. `dataset_manifest.csv` chứa toàn bộ dữ liệu hợp lệ. `experiments/pretrained_fluo/manifest.csv` dùng 20 ảnh validation và 20 ảnh test từ tập khởi đầu xám nền tối; giữ nguyên split. Phần train được giữ dự phòng. Mô hình đề xuất `2D_versatile_fluo` cần xác nhận miền ảnh bằng quan sát; cả mô hình này và `2D_paper_dsb2018` đã học một phần DSB2018 nên test cục bộ chưa phải kiểm thử tổng quát hóa độc lập. Đọc `HUONG_DAN_PRETRAINED.md`."),
             code("df = pd.read_csv(PROJECT / 'dataset_manifest.csv', keep_default_na=False)\ndisplay(df.groupby('split').agg(images=('source_id','count'), nuclei=('num_instances','sum')))\ndisplay(pd.crosstab(df['appearance_group'], df['split']))\nprimary = pd.read_csv(PROJECT / 'experiments/pretrained_fluo/manifest.csv', keep_default_na=False)\ndisplay(primary.groupby('split').agg(images=('source_id','count'), nuclei=('num_instances','sum')))\n"),
             md("## Kiểm tra ảnh, nhãn và chuẩn hóa\nMàu trong ảnh nhãn chỉ để hiển thị. TIFF lưu ID số nguyên. Ảnh xám nền tối có thêm `.npy` float32 theo công thức clip((I−P1)/(P99.8−P1),0,1). Không làm mượt hay chuẩn hóa nhãn."),
             code("display(Image.open(PROJECT / 'results/quality_control.png'))\ndisplay(Image.open(PROJECT / 'results/normalization_example.png'))\nr = primary[primary['split'] == 'test'].iloc[0]\nimage = np.asarray(Image.open(PROJECT / r.image_path))\nlabels = np.asarray(Image.open(PROJECT / r.mask_path))\nnormalized = np.load(PROJECT / r.normalized_path)\nprint('Original:', image.shape, image.dtype)\nprint('GT:', labels.shape, labels.dtype, 'instances:', len(np.unique(labels[labels > 0])))\nprint('Normalized:', normalized.shape, normalized.dtype, normalized.min(), normalized.max())\n"),
             md("## Kiểm tra gói dữ liệu\nKiểm tra kích thước, ID, hash ảnh gốc và phân chia; bước tái tạo ở trên kiểm tra trực tiếp từng mask nguồn."),
             code("sys.path.insert(0, str(PROJECT))\nfrom prepare_dataset import verify\nrecords = df.to_dict('records')\nprint(json.dumps(verify(PROJECT, records), indent=2))\n"),
             md("## Sử dụng và xuất dữ liệu\nCác thuật toán sử dụng cùng manifest và danh sách test. Chọn tham số trên validation, đánh giá cuối cùng trên test. Đọc `README.md`, `BAO_CAO_DU_LIEU.md` và `quality_report.json`. Random split không bảo đảm độc lập về thí nghiệm/nguồn hoặc với dữ liệu huấn luyện của mô hình pretrained.\n\nĐể nén lại sau khi thay đổi: chạy ô sau. ZIP nằm bên ngoài thư mục PROJECT, tránh tự nén chính nó."),
             code("REPACK = False\nif REPACK:\n    import shutil\n    archive = shutil.make_archive(str(PROJECT.parent / (PROJECT.name + '_handover')), 'zip', PROJECT.parent, PROJECT.name)\n    print(archive)\n")]
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                        language_info=dict(name="python", version="3.12")), nbformat=4, nbformat_minor=5)
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"
    (out / "Chuan_bi_du_lieu.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def documents(out, records, errors, raw, duplicate_groups):
    stats = summary_rows(records)
    write_csv(out / "dataset_summary.csv", stats)
    counts = Counter(r["appearance_group"] for r in records)
    total_nuclei = sum(r["num_instances"] for r in records)
    primary = [r for r in records if r["primary_100"]]
    table = "\n".join(f'| {r["split"]} | {r["num_images"]} | {r["num_instances"]} | {r["primary_100"]} |' for r in stats)
    group_text = "\n".join(f"- `{key}`: {value} ảnh." for key, value in sorted(counts.items()))
    readme = f"""# Bộ dữ liệu StarDist

Đã xử lý thực tế {len(records)} ảnh, {total_nuclei:,} nhân; {len(errors)} mẫu bị loại do kiểm tra dữ liệu. Nguồn gốc vẫn ở Downloads và không bị sửa. Chuẩn bị ngày 09/11/2026 (Asia/Saigon).

## Mở và sử dụng

**Chế độ pretrained:** không train/fine-tune. Tập thực nghiệm ban đầu là `experiments/pretrained_fluo/manifest.csv` (20 validation, 20 test, giữ nguyên split); mô hình đề xuất `2D_versatile_fluo`. Phần train được giữ dự phòng. Xem `HUONG_DAN_PRETRAINED.md` và `experiments/pretrained_fluo/config.json`. Cả `2D_versatile_fluo` lẫn `2D_paper_dsb2018` đã học một phần DSB2018 theo tài liệu StarDist, nên bản test này chưa chứng minh đánh giá tổng quát hóa độc lập.

1. Giải nén ZIP vào một thư mục; mở `Chuan_bi_du_lieu.ipynb` trên Jupyter/Colab, sửa `PROJECT` tới thư mục này.
2. Các thuật toán sử dụng cùng manifest. Dùng cột `image_path`, `mask_path`; đường dẫn tương đối với thư mục này.
3. TIFF mask là **uint16**, nền = 0, mỗi nhân = một ID 1..N. Đọc bằng `np.asarray(Image.open(path))` hoặc `tifffile.imread(path)`; không convert mask sang uint8 hoặc nhị phân khi tính matching instance.
4. `experiments/grayscale100/manifest.csv` có {len(primary)} ảnh khởi đầu; `train_ids.txt`, `val_ids.txt`, `test_ids.txt` là các source ID dùng chung. Đó là các dòng tham chiếu dữ liệu, không nhân bản tệp.
5. Ảnh PNG gốc giữ nguyên byte và kích thước. Cột `normalized_path` trỏ đến NPY float32 đơn kênh cho nhóm ảnh xám nền tối; cột trống nghĩa là chưa áp dụng chuẩn hóa này.

## Phân chia toàn bộ dữ liệu

| Tập | Ảnh | Nhân | Ảnh trong tập khởi đầu |
|---|---:|---:|---:|
{table}

Seed = 42; phân tầng theo diện mạo và số nhân (<20, 20–59, >=60). Giữ ảnh có cùng RGB pixel hash trong một tập. Có {duplicate_groups} nhóm ảnh trùng pixel. Tỉ lệ mục tiêu 60/20/20, làm tròn theo từng tầng/nhóm. Chia ở cấp ảnh, trước khi tạo patch/augmentation. Không dùng test để chọn tham số. Nếu chỉ dùng pretrained, phần train có thể chưa dùng.

## Nhóm diện mạo

{group_text}

`grayscale_*`: ba kênh RGB bằng nhau ở mọi pixel (bao gồm ảnh xám lưu dưới dạng RGB/RGBA). Nền tối/sáng được gợi ý bằng median dải biên 5%, ngưỡng 127.5 trên thang 0..255. Đây là **quy tắc diện mạo**, không xác nhận huỳnh quang hoặc H&E. Không coi mọi ảnh màu là H&E. Alpha gốc được giữ trong PNG; khi phân loại/hash dùng RGB, không dùng alpha.

Tập khởi đầu lấy ngẫu nhiên tối đa 60/20/20 ảnh từ nhóm xám nền tối của từng split, không chọn bằng chất lượng dự đoán. Cần xác nhận mô hình StarDist và kiểm tra trực quan ảnh trước khi chốt thí nghiệm. Các nhóm diện mạo khác được giữ trong manifest đầy đủ.

## Tiền xử lý

Không resize/crop, đổi hướng, Gaussian, CLAHE hoặc chỉnh sửa nhãn. Ảnh gốc được giữ để thử tiền xử lý riêng cho Otsu/Watershed trên validation.

Với nhóm xám nền tối: lấy kênh R (R=G=B), tính P1 và P99.8 trên từng ảnh, `clip((I-P1)/(P99.8-P1),0,1)` và lưu NPY float32. Tham số nằm trong manifest; không tính percentile chung trên test/train. Đây là lựa chọn được ghi rõ của pipeline này; không mặc định trùng mọi cấu hình chuẩn hóa StarDist. Không chuẩn hóa hai lần. Các nhóm khác cần pipeline riêng phù hợp mô hình.

## Kiểm tra và truy xuất

- `quality_report.json`: kết quả kiểm tra số lượng, kích thước, giá trị mask, chồng lấn, TIFF đọc lại, hash ảnh và chuẩn hóa.
- `instance_mapping.csv`: truy từ mỗi instance ID về tên mask nguồn và diện tích.
- `source_inventory.csv`: thống kê và lý do loại, gồm tất cả thư mục nguồn được phát hiện.
- `results/quality_control.png`, `results/normalization_example.png`: hình minh họa; `preview_samples.csv` ghi đầy đủ ID các ảnh xem.
- `checksums.sha256`: SHA256 các tệp của gói (trừ chính tệp checksum).
- `prepare_dataset.py`: mã tái tạo; chỉ cần numpy và Pillow. Script từ chối ghi đè thư mục đầu ra đã có nội dung.

Chạy lại ở Windows/Colab:
```bash
python -m pip install -r requirements.txt
python prepare_dataset.py --raw "PATH_TO/stage1_train" --out "PATH_TO/new_output"
```
`--raw` nhận thư mục chứa các thư mục image ID hoặc file `stage1_train.zip`. ZIP bàn giao chỉ chứa ảnh gốc được sao chép, nhãn gộp và tài liệu; không kèm lại hàng nghìn mask nguồn. Nguồn chạy thực tế: `{raw}`.

## Giới hạn cần ghi trong báo cáo

Không có metadata nguồn thí nghiệm được tích hợp vào bản chia này, nên chưa kiểm soát nhóm bệnh nhân/đĩa/nguồn thu nhận hoặc ảnh gần trùng. Hash chỉ loại rò rỉ ảnh trùng pixel; chia ngẫu nhiên không chứng minh tính độc lập sinh học. Một số pretrained StarDist có liên quan DSB2018: đánh giá trên đây không tự động là đánh giá tổng quát hóa độc lập. Nhãn gốc có thể có thiếu sót chú giải; chưa sửa nhãn bằng mô hình. Gói dữ liệu chưa chứa kết quả dự đoán hoặc điểm đánh giá phân đoạn.

## Thành phần sử dụng

- **StarDist:** ảnh, TIFF instance và danh sách ID; chọn mô hình và pipeline phù hợp miền ảnh.
- **Otsu/Watershed:** cùng danh sách val/test; chỉnh tham số trên validation và lưu kết quả ở cấp instance để so sánh.
- **Báo cáo:** `BAO_CAO_DU_LIEU.md`, bảng thống kê và hình minh họa.

## Nguồn

- BBBC038v1: https://bbbc.broadinstitute.org/BBBC038 (CC0; mô tả cấu trúc ảnh và mask).
- Caicedo et al. (2019), *Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl*, https://doi.org/10.1038/s41592-019-0612-7.
- StarDist: https://github.com/stardist/stardist ; danh sách mô hình https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    report = f"""# Dữ liệu và tiền xử lý

## 3.1. Giới thiệu bộ dữ liệu

Thí nghiệm sử dụng gói stage1_train của Data Science Bowl 2018, được phân phối qua BBBC038v1 [1]. Bộ dữ liệu gồm ảnh nhân tế bào từ nhiều điều kiện thu nhận; mỗi nhân được chú giải bằng một mask nhị phân riêng. Pipeline phát hiện {len(records) + len(errors)} thư mục mẫu, giữ {len(records)} ảnh hợp lệ, tổng cộng {total_nuclei:,} nhân; {len(errors)} mẫu bị loại. Số liệu được tính trực tiếp từ tệp cục bộ.

## 3.2. Cấu trúc và nhãn

Ảnh PNG được sao chép nguyên byte, giữ kích thước gốc, từ {min(r['width'] for r in records)}–{max(r['width'] for r in records)} pixel chiều rộng và {min(r['height'] for r in records)}–{max(r['height'] for r in records)} pixel chiều cao. Các mask nguồn được sắp xếp theo tên, kiểm tra rồi gộp thành một label map TIFF uint16: nền 0 và ID 1..N cho từng nhân. Không gộp các nhân thành nhãn nhị phân. Mỗi nhãn có bản ghi truy xuất về mask PNG nguồn trong instance_mapping.csv. Pipeline phát hiện mask sai kích thước, rỗng, không nhị phân hoặc chồng lấn và loại cả mẫu đó kèm lý do thay vì âm thầm ghi đè.

## 3.3. Phân chia dữ liệu

| Tập | Số ảnh | Số nhân | Tập khởi đầu |
|---|---:|---:|---:|
{table}

Chia train/validation/test theo mục tiêu 60/20/20, seed 42, phân tầng theo nhóm diện mạo và mức số lượng nhân. Ảnh trùng chính xác RGB pixel được gom vào cùng một tập ({duplicate_groups} nhóm trùng). Tỉ lệ thực tế chịu ảnh hưởng làm tròn và kích thước nhóm. Validation phục vụ chỉnh tham số; test dùng cho đánh giá cuối cùng. Các thuật toán dùng cùng source ID để phép so sánh nhất quán. Nếu tạo patch/augmentation, phải thực hiện sau khi chia tập và chỉ augment train.

Tập khởi đầu gồm {len(primary)} ảnh xám nền tối trong experiments/grayscale100/manifest.csv (mục tiêu 60 train, 20 val, 20 test). Các ảnh được lấy ngẫu nhiên trong từng split, chưa xác nhận miền nhuộm và chưa dựa trên điểm dự đoán. Quy mô thực nghiệm được xác định theo manifest sử dụng.

**Cấu hình thực nghiệm: pretrained với trọng số cố định, không huấn luyện/fine-tune.** Thí nghiệm khởi đầu dùng 20 ảnh validation và 20 ảnh test trong experiments/pretrained_fluo/manifest.csv, lấy nguyên các split của tập grayscale100; 60 ảnh train được giữ dự phòng. Mô hình đề xuất là 2D_versatile_fluo sau khi xác nhận miền ảnh. Cả 2D_versatile_fluo và 2D_paper_dsb2018 được StarDist mô tả là đã huấn luyện trên một phần DSB2018 [2]; do chưa đối chiếu danh sách huấn luyện gốc, không diễn giải kết quả là đánh giá tổng quát hóa độc lập.

## 3.4. Tiền xử lý

Các nhóm diện mạo được thống kê:

{group_text}

Quy tắc ảnh xám yêu cầu R=G=B tại mọi pixel; nền được gợi ý từ median dải biên 5% với ngưỡng 127.5. Quy tắc này không thay thế thông tin loại kính hiển vi hoặc nhuộm. Với ảnh xám nền tối, xuất thêm ảnh đơn kênh float32 theo I_norm = clip((I−P1)/(P99.8−P1),0,1); percentile được tính riêng từng ảnh và ghi trong manifest. Ảnh PNG gốc luôn giữ nguyên. Không resize, làm mượt hoặc áp biến đổi cường độ lên nhãn. Otsu/Watershed có thể cần tiền xử lý riêng, được chọn trên validation và ghi rõ trong phần thực nghiệm.

## 3.5. Kiểm tra chất lượng và giới hạn

Kiểm tra tự động bao gồm cấu trúc ảnh/mask, kích thước, mask nhị phân không rỗng, chồng lấn, đọc lại TIFF, ID liên tục và diện tích khớp từng mask nguồn, SHA256 bản sao PNG, phạm vi NPY chuẩn hóa và ảnh trùng qua các split. Minh họa original/instance label/biên nhãn nằm trong results/quality_control.png; thông tin mẫu trong results/preview_samples.csv. Hình chuẩn hóa nằm trong results/normalization_example.png. Kết quả chi tiết trong quality_report.json.

Bản chia ở cấp ảnh chưa sử dụng nhóm nguồn thí nghiệm nên không loại trừ tương quan giữa các ảnh hoặc ảnh gần trùng. Việc kiểm tra cơ học không xác nhận tính đúng đắn sinh học của mọi chú giải. Không dùng dự đoán làm ground truth. Với mô hình pretrained, cần kiểm tra sự giao nhau với dữ liệu huấn luyện trước khi tuyên bố đánh giá độc lập; StarDist có mô hình mang tên 2D_paper_dsb2018 [2]. Phần này mới mô tả chuẩn bị dữ liệu, chưa có kết quả so sánh StarDist/Otsu/Watershed.

## Tài liệu tham khảo

[1] BBBC038v1, Broad Bioimage Benchmark Collection. https://bbbc.broadinstitute.org/BBBC038 ; Caicedo et al. (2019), Nature Methods. https://doi.org/10.1038/s41592-019-0612-7.

[2] StarDist, mã nguồn và danh sách mô hình pretrained. https://github.com/stardist/stardist ; https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py.
"""
    (out / "BAO_CAO_DU_LIEU.md").write_text(report, encoding="utf-8")
    (out / "requirements.txt").write_text("numpy>=1.26,<3\nPillow>=10,<13\n", encoding="utf-8")


def prepare(raw, out):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    if not raw.exists():
        raise ValueError(f"Missing source: {raw}")
    if raw == out or raw in out.parents or out in raw.parents:
        raise ValueError("Source and output directories must be separate")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output is not empty. Choose a new output directory.")
    archive = None
    if raw.is_file() and raw.suffix.lower() == '.zip':
        # Open archive from memory to avoid tens of thousands of small filesystem reads.
        archive = zipfile.ZipFile(io.BytesIO(raw.read_bytes()))
        source_root = zipfile.Path(archive)
        children = [p for p in source_root.iterdir() if p.is_dir()]
        if len(children) == 1 and not (children[0] / 'images').is_dir():
            source_root = children[0]
    elif raw.is_dir():
        source_root = raw
    else:
        raise ValueError('Source must be an extracted directory or ZIP')
    samples = sorted((p for p in source_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not samples:
        raise ValueError("No sample directories")
    out.mkdir(parents=True, exist_ok=True)
    staging = out / "_labels_staging"
    staging.mkdir()
    records, errors, mapping, inventory = [], [], [], []
    for i, sample in enumerate(samples):
        try:
            files = png_files(sample / "images")
            if len(files) != 1 or not (sample / "masks").is_dir():
                raise ValueError("Expected one image and a masks directory")
            image, rgb, mode = load_image(files[0])
            if image.dtype != np.uint8:
                raise ValueError(f"Unsupported image dtype: {image.dtype}")
            if image.ndim == 3 and image.shape[-1] == 4 and not np.all(image[..., 3] == 255):
                raise ValueError("Non-opaque alpha needs explicit handling")
            h, w = rgb.shape[:2]
            labels, instance_rows = merge_masks(sample, (h, w))
            group, gray, border_median = appearance(rgb)
            norm_path = ""
            p1, p998 = "", ""
            if group == PROFILE:
                _, p1, p998 = normalize_gray(gray)
            label_path = staging / (sample.name + ".tif")
            Image.fromarray(labels).save(label_path, compression="raw")
            reread = np.asarray(Image.open(label_path))
            if not np.array_equal(labels, reread):
                raise ValueError("TIFF roundtrip differs from merged source masks")
            # Every source mask's exact pixels were inserted without overwrite;
            # check output area and label IDs against the original per-mask areas.
            areas = np.bincount(reread.ravel(), minlength=len(instance_rows) + 1)
            if any(int(areas[m["instance_id"]]) != m["area_pixels"] for m in instance_rows):
                raise ValueError("Instance area does not match original mask")
            digest = hashlib.sha256(np.asarray(rgb.shape, dtype=np.int64).tobytes() + rgb.tobytes()).hexdigest()
            records.append(dict(source_id=sample.name, split="", image_path="", mask_path="",
                                normalized_path=norm_path, width=w, height=h, source_mode=mode,
                                appearance_group=group, border_median=border_median,
                                num_instances=len(instance_rows), foreground_pixels=int((labels > 0).sum()),
                                source_file_sha256=sha256(files[0]), pixel_sha256=digest,
                                p1=p1, p99_8=p998, primary_100=0))
            mapping.extend(instance_rows)
            inventory.append(dict(source_id=sample.name, status="accepted", reason=""))
        except (ValueError, OSError) as exc:
            errors.append(dict(source_id=sample.name, reason=str(exc)))
            inventory.append(dict(source_id=sample.name, status="excluded", reason=str(exc)))
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"Audited {i+1}/{len(samples)} samples; valid={len(records)}, excluded={len(errors)}", flush=True)
    write_csv(out / "source_inventory.csv", inventory)
    if not records:
        raise ValueError("No valid records. See source_inventory.csv")
    split_records(records)
    for r in records:
        base = Path("processed_data") / r["split"]
        stem = r["source_id"]
        for folder in ["images", "masks", "normalized"]:
            (out / base / folder).mkdir(parents=True, exist_ok=True)
        r["image_path"] = (base / "images" / (stem + ".png")).as_posix()
        r["mask_path"] = (base / "masks" / (stem + ".tif")).as_posix()
        source_image = png_files(source_root / stem / "images")[0]
        with source_image.open('rb') as f, (out / r["image_path"]).open('wb') as target:
            shutil.copyfileobj(f, target)
        shutil.move(str(staging / (stem + ".tif")), out / r["mask_path"])
        if r["appearance_group"] == PROFILE:
            _, rgb, _ = load_image(source_image)
            norm, _, _ = normalize_gray(rgb[..., 0])
            r["normalized_path"] = (base / "normalized" / (stem + ".npy")).as_posix()
            np.save(out / r["normalized_path"], norm, allow_pickle=False)
    # Only this known staging directory, created in the output above, is removed if empty.
    for remaining in staging.iterdir():
        if remaining.is_file() and remaining.suffix == ".tif":
            remaining.unlink()
    staging.rmdir()
    if archive is not None:
        archive.close()
    write_csv(out / "dataset_manifest.csv", records)
    write_csv(out / "instance_mapping.csv", mapping)
    for folder, selected in [(out / "splits", records),
                              (out / "experiments/grayscale100", [r for r in records if r["primary_100"]])]:
        folder.mkdir(parents=True)
        write_csv(folder / "manifest.csv", selected)
        for split in ["train", "val", "test"]:
            ids = sorted(r["source_id"] for r in selected if r["split"] == split)
            (folder / f"{split}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    duplicate_groups = sum(v > 1 for v in Counter(r["pixel_sha256"] for r in records).values())
    print("Verifying all exported pairs...", flush=True)
    quality = verify(out, records)
    quality.update(discovered_samples=len(samples), excluded_samples=errors,
                   masks_checked=len(mapping), overlapping_masks_in_accepted=0,
                   empty_masks_in_accepted=0, duplicate_pixel_groups=duplicate_groups,
                   appearance_counts=dict(Counter(r["appearance_group"] for r in records)),
                   seed=SEED, split_target=[.6, .2, .2],
                   limitations=["No acquisition/source metadata grouping", "Near-duplicates not checked",
                                "Pretrained training-set overlap not established", "Appearance is not verified modality"])
    quality['source'] = str(raw)
    if raw.is_file():
        quality['source_archive_sha256'] = sha256(raw)
    quality['runtime'] = dict(python=sys.version.split()[0], numpy=np.__version__, pillow=Image.__version__)
    (out / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    make_previews(out, records)
    pretrained_profile(out, records)
    documents(out, records, errors, raw, duplicate_groups)
    notebook(out)
    script = Path(__file__).resolve()
    if script != out / script.name:
        shutil.copy2(script, out / script.name)
    checksum_lines = [f"{sha256(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*"))
                      if p.is_file() and p.name != "checksums.sha256"]
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False, indent=2), flush=True)
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--zip", action="store_true", help="Create and CRC-check a handover ZIP next to output")
    args = parser.parse_args()
    prepare(args.raw, args.out)
    if args.zip:
        out = args.out.resolve()
        archive = shutil.make_archive(str(out.parent / (out.name + "_BanGiao")), "zip", out.parent, out.name)
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failure: {bad}")
        print(f"Archive verified: {archive}", flush=True)


if __name__ == "__main__":
    main()
