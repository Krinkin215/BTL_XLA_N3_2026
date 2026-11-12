"""Reproducible StarDist/Otsu/Watershed experiment; tune on validation only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# These must be set before TensorFlow/CSBDeep is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))

import numpy as np
from PIL import Image

IOU_THRESHOLDS = tuple(round(x / 100, 2) for x in range(50, 100, 5))
BASELINE_GRID = {"sigma": [0.8, 1.0, 1.5], "min_size": [20, 30, 50]}
NMS_GRID = [0.3, 0.4, 0.5]
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty table")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_path(root, value):
    path = (root / value).resolve()
    if not value or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Invalid dataset path: {value!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_manifest(root, manifest):
    """Require existing splits and source paths from the prepared master manifest."""
    with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (root / "dataset_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        master = {r["source_id"]: r for r in csv.DictReader(stream)}
    required = {"source_id", "split", "image_path", "mask_path", "normalized_path",
                "width", "height", "num_instances"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain rows and columns {sorted(required)}")
    ids = [r["source_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source_id in experiment manifest")
    if {r["split"] for r in rows} != {"val", "test"}:
        raise ValueError("Experiment must contain only val and test, both nonempty")
    for row in rows:
        sid = row["source_id"]
        if not sid or Path(sid).name != sid or sid in {".", ".."}:
            raise ValueError(f"Invalid source_id: {sid!r}")
        if sid not in master or any(row[k] != master[sid][k] for k in required):
            raise ValueError(f"Split or source metadata differs from master: {sid}")
        if master[sid].get("appearance_group") != "grayscale_dark_background":
            raise ValueError(f"This fluorescent single-channel protocol requires dark grayscale input: {sid}")
        for key in ("image_path", "mask_path", "normalized_path"):
            dataset_path(root, row[key])
    return sorted(rows, key=lambda r: (r["split"], r["source_id"]))


def load_split(root, rows, split):
    selected = [r for r in rows if r["split"] == split]
    images, truths = [], []
    for row in selected:
        image = np.load(dataset_path(root, row["normalized_path"]), allow_pickle=False)
        with Image.open(dataset_path(root, row["mask_path"])) as source:
            truth = np.array(source)
        shape = (int(row["height"]), int(row["width"]))
        if image.dtype != np.float32 or image.ndim != 2 or image.shape != shape:
            raise ValueError(f"Expected normalized float32 HxW input: {row['source_id']}")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"Normalized input must be finite in [0,1]: {row['source_id']}")
        if truth.shape != shape or not np.issubdtype(truth.dtype, np.integer) or truth.min() < 0:
            raise ValueError(f"Invalid integer instance mask: {row['source_id']}")
        if np.count_nonzero(np.unique(truth)) != int(row["num_instances"]):
            raise ValueError(f"Ground truth instance count differs from manifest: {row['source_id']}")
        images.append(image)
        truths.append(truth)
    return selected, images, truths


def otsu_seg(image, sigma=1.0, min_size=30):
    from skimage.filters import gaussian, threshold_otsu
    from skimage.measure import label
    from skimage.morphology import binary_opening, disk, remove_small_objects

    smoothed = gaussian(image, sigma=sigma, preserve_range=True)
    if np.ptp(smoothed) <= 1e-8:
        return np.zeros(image.shape, np.int32)
    foreground = binary_opening(smoothed > threshold_otsu(smoothed), disk(2))
    foreground = remove_small_objects(foreground, min_size=min_size)
    return label(foreground).astype(np.int32)


def watershed_stages(image, sigma=1.0, min_size=30, min_distance=5):
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    components = otsu_seg(image, sigma=sigma, min_size=min_size)
    foreground = components > 0
    distance = ndi.distance_transform_edt(foreground)
    markers = np.zeros(image.shape, np.int32)
    if np.any(foreground):
        coords = peak_local_max(distance, min_distance=min_distance,
                                labels=components, exclude_border=False)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        # Every component needs a seed, including small or border components.
        seeded = set(np.unique(components[markers > 0]))
        next_marker = int(markers.max()) + 1
        for component in np.unique(components[foreground]):
            if component not in seeded:
                y, x = np.unravel_index(np.argmax(np.where(components == component, distance, -1)), image.shape)
                markers[y, x] = next_marker
                next_marker += 1
    prediction = watershed(-distance, markers, mask=foreground).astype(np.int32)
    return foreground, distance, markers, prediction


def watershed_seg(image, **config):
    return watershed_stages(image, **config)[-1]


def score_instances(truths, predictions, thresholds=IOU_THRESHOLDS):
    from stardist.matching import matching_dataset

    stats = matching_dataset(truths, predictions, thresh=list(thresholds),
                             by_image=False, show_progress=False)
    return [{"iou_threshold": float(s.thresh), "tp": int(s.tp), "fp": int(s.fp),
             "fn": int(s.fn), "n_true": int(s.n_true), "n_pred": int(s.n_pred),
             "precision": float(s.precision), "recall": float(s.recall),
             "f1": float(s.f1), "matching_accuracy": float(s.accuracy),
             "mean_matched_iou": float(s.mean_matched_score),
             "panoptic_quality": float(s.panoptic_quality)} for s in stats]


def tune_baseline(method, images, truths):
    keys = ["sigma", "min_size"] + (["min_distance"] if method == "watershed" else [])
    values = [BASELINE_GRID[k] if k in BASELINE_GRID else [3, 5, 7, 9] for k in keys]
    predict = otsu_seg if method == "otsu" else watershed_seg
    trials = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        predictions = [predict(image, **config) for image in images]
        score = score_instances(truths, predictions, (0.5,))[0]["matching_accuracy"]
        trials.append({"method": method, **config, "val_matching_accuracy_at_0_5": score})
    # Stable first-in-grid tie breaking, with no access to test predictions.
    best = max(trials, key=lambda r: r["val_matching_accuracy_at_0_5"])
    return {k: best[k] for k in keys}, trials


def save_prediction(path, prediction, shape):
    prediction = np.asarray(prediction)
    if prediction.shape != shape or not np.issubdtype(prediction.dtype, np.integer) or prediction.min() < 0:
        raise ValueError("Prediction must be a non-negative integer instance map matching input shape")
    if prediction.max() > np.iinfo(np.uint32).max:
        raise ValueError("Prediction label exceeds uint32")
    dtype = np.uint16 if prediction.max() <= np.iinfo(np.uint16).max else np.uint32
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, prediction.astype(dtype), photometric="minisblack", compression="zlib")


def load_pretrained_model():
    """Use the official registry/hash with an explicitly writable local cache."""
    from csbdeep.models.pretrained import get_model_details
    from csbdeep.utils.tf import keras_import
    from stardist.models import StarDist2D

    key, _, details = get_model_details(StarDist2D, "2D_versatile_fluo")
    cache = Path(os.environ["KERAS_HOME"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive = Path(keras_import("utils", "get_file")(
        fname=key + ".zip", origin=details["url"], file_hash=details["hash"],
        cache_dir=str(cache), cache_subdir=f"models/StarDist2D/{key}", extract=True))
    model = StarDist2D(config=None, name=key, basedir=archive.parent.parent)
    model.basedir = None  # Treat cached pretrained weights as read-only, like from_pretrained.
    return model


def evaluate(method, predict, split, rows, images, truths, output):
    from stardist.matching import matching

    predictions, per_image, files = [], [], []
    for row, image, truth in zip(rows, images, truths):
        start = time.perf_counter()
        prediction = predict(image)
        elapsed = time.perf_counter() - start
        destination = output / "predictions" / method / split / f"{row['source_id']}.tif"
        save_prediction(destination, prediction, image.shape)
        predictions.append(prediction)
        stats = matching(truth, prediction, thresh=0.5)
        per_image.append({"method": method, "split": split, "source_id": row["source_id"],
                          "iou_threshold": 0.5, "tp": int(stats.tp), "fp": int(stats.fp),
                          "fn": int(stats.fn), "n_true": int(stats.n_true), "n_pred": int(stats.n_pred),
                          "precision": float(stats.precision), "recall": float(stats.recall),
                          "f1": float(stats.f1), "matching_accuracy": float(stats.accuracy),
                          "inference_seconds": elapsed})
        files.append({"method": method, "split": split, "source_id": row["source_id"],
                      "image_path": row["image_path"], "mask_path": row["mask_path"],
                      "prediction_path": destination.relative_to(output).as_posix(),
                      "prediction_sha256": sha256(destination)})
    mean_seconds = sum(r["inference_seconds"] for r in per_image) / len(rows)
    summary = [{"method": method, "split": split, "images": len(rows), **s,
                "mean_inference_seconds": mean_seconds} for s in score_instances(truths, predictions)]
    print(f"{split} / {method}: F1@0.5={summary[0]['f1']:.4f}, "
          f"matching_accuracy@0.5={summary[0]['matching_accuracy']:.4f}", flush=True)
    return predictions, summary, per_image, files


def render_instances(labels, image):
    """Overlay instance colors without importing legacy StarDist plotting APIs."""
    unique, inverse = np.unique(labels, return_inverse=True)
    palette = np.random.default_rng(SEED).uniform(0.25, 1.0, size=(len(unique), 3))
    colors = palette[inverse].reshape((*image.shape, 3))
    overlay = np.repeat(image[..., None], 3, axis=-1)
    foreground = labels > 0
    overlay[foreground] = 0.35 * overlay[foreground] + 0.65 * colors[foreground]
    return overlay


def export_figures(output, rows, images, truths, predictions, summary, model, configs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir()
    order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["num_instances"]), rows[i]["source_id"]))
    indices = list(dict.fromkeys([order[0], order[len(order) // 2], order[-1]]))
    methods = list(predictions)
    titles = {"otsu": "Otsu", "watershed": "Otsu + Watershed", "stardist": "StarDist"}
    fig, axes = plt.subplots(len(indices), 2 + len(methods), figsize=(4 * (2 + len(methods)), 4 * len(indices)), squeeze=False)
    for r, i in enumerate(indices):
        axes[r, 0].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"Input {rows[i]['source_id'][:8]}\nGT: {rows[i]['num_instances']} nuclei")
        axes[r, 1].imshow(render_instances(truths[i], images[i]))
        axes[r, 1].set_title("Ground truth")
        for c, method in enumerate(methods, 2):
            pred = predictions[method][i]
            axes[r, c].imshow(render_instances(pred, images[i]))
            axes[r, c].set_title(f"{titles[method]}: {np.count_nonzero(np.unique(pred))}")
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "segmentation_comparison.png", dpi=160)
    plt.close(fig)
    write_json(figure_dir / "sample_ids.json", [rows[i]["source_id"] for i in indices])

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        scores = [r for r in summary if r["split"] == "test" and r["method"] == method]
        ax.plot([r["iou_threshold"] for r in scores], [r["matching_accuracy"] for r in scores], "o-", label=titles[method])
    ax.set(xlabel="Matching IoU threshold", ylabel="TP / (TP + FP + FN)", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "matching_accuracy_iou.png", dpi=160)
    plt.close(fig)

    i = indices[len(indices) // 2]
    probability, _ = model.predict(images[i], axes="YX", verbose=0)
    np.save(figure_dir / "stardist_probability.npy", probability)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(images[i], cmap="gray")
    axes[0].set_title(f"Input {rows[i]['source_id'][:8]}")
    axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Probability (model grid)")
    axes[2].imshow(render_instances(predictions["stardist"][i], images[i]))
    axes[2].set_title("StarDist after NMS")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "stardist_probability.png", dpi=160)
    plt.close(fig)
    if "watershed" in configs:
        stages = watershed_stages(images[i], **configs["watershed"])
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for ax, value, title in zip(axes, [images[i], *stages],
                                    ["Input", "Otsu foreground", "Distance transform", "Markers", "Watershed"]):
            ax.imshow(value, cmap="gray" if title in {"Input", "Otsu foreground"} else "nipy_spectral")
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(figure_dir / "watershed_steps.png", dpi=160)
        plt.close(fig)


def export_report(output, summary, configs):
    test = [r for r in summary if r["split"] == "test" and r["iou_threshold"] == 0.5]
    lines = ["# Kết quả chạy thực nghiệm", "", "Số liệu được sinh trực tiếp bởi `run_experiment.py` trong lần chạy này.", "",
             "| Phương pháp | Ảnh test | TP | FP | FN | Precision | Recall | F1 | Matching accuracy | Giây/ảnh |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in test:
        lines.append(f"| {row['method']} | {row['images']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                     f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
                     f"{row['matching_accuracy']:.4f} | {row['mean_inference_seconds']:.4f} |")
    star = next(r for r in test if r["method"] == "stardist")
    lines += ["", "IoU matching = 0.5; cộng TP/FP/FN trên toàn bộ tập trước khi tính các chỉ số (micro).",
              "Matching accuracy = TP/(TP+FP+FN); đây không phải COCO AP hoặc diện tích dưới đường precision–recall.", "",
              f"StarDist đạt F1 = {star['f1']:.4f}, precision = {star['precision']:.4f}, recall = {star['recall']:.4f}."]
    baselines = [r for r in test if r["method"] != "stardist"]
    if baselines:
        best = max(baselines, key=lambda r: r["f1"])
        lines.append(f"Chênh lệch F1 với baseline cao nhất ({best['method']}): {star['f1'] - best['f1']:+.4f}.")
    lines += ["", "Cấu hình đã chốt trên validation trước khi nạp nhãn test:", "", "```json",
              json.dumps(configs, ensure_ascii=False, indent=2), "```", "",
              "Thời gian chỉ gồm suy luận và hậu xử lý; đã warm-up trên một ảnh validation. Không gồm tải mô hình, "
              "tuning, đọc/ghi tệp, matching hoặc vẽ hình. Phụ thuộc phần cứng và kích thước ảnh.", "",
              "Đây là so sánh trong tập DSB2018 đã chọn. Mô hình pretrained đã học một phần DSB2018; "
              "chưa xác minh giao nhau với tập test, nên chưa thể kết luận khả năng tổng quát hóa độc lập.", "",
              "Chi tiết: `metrics_summary.csv`, `metrics_per_image.csv`, `predictions_manifest.csv`, "
              "`selected_config.json`, `run_metadata.json`, `figures/`.", ""]
    (output / "KET_QUA.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(dataset, manifest, output, *, include_baselines=True, tune_stardist=True):
    dataset, manifest, output = Path(dataset).resolve(), Path(manifest).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty; choose a new run directory: {output}")
    rows = read_manifest(dataset, manifest)
    val_rows, x_val, y_val = load_split(dataset, rows, "val")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "experiment_manifest.csv", rows)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    model = load_pretrained_model()
    defaults = model.thresholds._asdict()
    print(f"Tuning on {len(x_val)} validation images only", flush=True)
    if tune_stardist:
        model.optimize_thresholds(x_val, y_val, nms_threshs=NMS_GRID, iou_threshs=[0.5],
                                  predict_kwargs={"axes": "YX", "verbose": 0}, save_to_json=False)
    thresholds = {"prob_thresh": float(model.thresholds.prob), "nms_thresh": float(model.thresholds.nms)}
    configs = {"stardist": thresholds}
    if include_baselines:
        for method in ("otsu", "watershed"):
            configs[method], trials = tune_baseline(method, x_val, y_val)
            write_csv(output / f"validation_search_{method}.csv", trials)
    selected = {"model": "2D_versatile_fluo", "seed": SEED, "training": False, "fine_tuning": False,
                "selection_split": "val", "selection_iou_thresholds": [0.5],
                "selection_metric": "matching_accuracy", "stardist_tuned": tune_stardist,
                "nms_grid": NMS_GRID, "model_default_thresholds": defaults, "methods": configs,
                "validation_ids": [r["source_id"] for r in val_rows],
                "frozen_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()}
    # Freeze configuration before test images or ground truth are loaded.
    write_json(output / "selected_config.json", selected)
    frozen_sha = sha256(output / "selected_config.json")
    predict = {"stardist": lambda image: model.predict_instances(image, axes="YX", verbose=0, **thresholds)[0]}
    if include_baselines:
        predict.update(otsu=lambda image: otsu_seg(image, **configs["otsu"]),
                       watershed=lambda image: watershed_seg(image, **configs["watershed"]))
    for predict_fn in predict.values():
        predict_fn(x_val[0])
    summary, per_image, prediction_files = [], [], []
    for split in ("val", "test"):
        split_rows, images, truths = (val_rows, x_val, y_val) if split == "val" else load_split(dataset, rows, "test")
        split_predictions = {}
        for method, predict_fn in predict.items():
            preds, scores, individual, files = evaluate(method, predict_fn, split, split_rows, images, truths, output)
            split_predictions[method] = preds
            summary.extend(scores)
            per_image.extend(individual)
            prediction_files.extend(files)
        if split == "test":
            export_figures(output, split_rows, images, truths, split_predictions, summary, model, configs)
    if sha256(output / "selected_config.json") != frozen_sha:
        raise RuntimeError("Frozen configuration changed during evaluation")
    write_csv(output / "metrics_summary.csv", summary)
    write_csv(output / "metrics_per_image.csv", per_image)
    write_csv(output / "predictions_manifest.csv", prediction_files)
    packages = ["numpy", "Pillow", "tensorflow", "tf-keras", "stardist", "csbdeep", "scikit-image",
                "scipy", "matplotlib", "tifffile"]
    model_files = {p.name: sha256(p) for p in Path(model.logdir).glob("*") if p.is_file()}
    provenance = [{"source_id": r["source_id"], "split": r["split"],
                   **{k + "_sha256": sha256(dataset_path(dataset, r[k]))
                      for k in ("normalized_path", "mask_path")}} for r in rows]
    write_json(output / "run_metadata.json", {
        "status": "complete", "completed_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "tensorflow_devices": [d.name for d in tf.config.list_physical_devices()],
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "script_sha256": sha256(__file__), "manifest_sha256": sha256(manifest),
        "selected_config_sha256": frozen_sha, "pretrained_model_files_sha256": model_files,
        "inputs": provenance, "aggregation": "micro", "normalization": "existing P1/P99.8 NPY; no second normalization",
        "independent_generalization_evaluation": False})
    export_report(output, summary, configs)
    checksums = [f"{sha256(p)}  {p.relative_to(output).as_posix()}" for p in sorted(output.rglob("*")) if p.is_file()]
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"Completed. Results: {output / 'KET_QUA.md'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "StarDist_Dataset")
    parser.add_argument("--manifest", type=Path, default=Path("experiments/pretrained_fluo/manifest.csv"),
                        help="Absolute path, or relative to --dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/stardist_pretrained")
    parser.add_argument("--stardist-only", action="store_true", help="Skip the two comparison baselines")
    parser.add_argument("--default-thresholds", action="store_true", help="Use pretrained thresholds without validation tuning")
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else args.dataset / args.manifest
    try:
        run_experiment(args.dataset, manifest, args.output, include_baselines=not args.stardist_only,
                       tune_stardist=not args.default_thresholds)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Error: {error}\nInstall dependencies: python -m pip install -r requirements-stardist.txt\n")


if __name__ == "__main__":
    main()
