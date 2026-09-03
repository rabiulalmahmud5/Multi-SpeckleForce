"""Reproduce the Multi-SpeckleForce kNN and Random-Forest baselines.

Dataset Available at https://doi.org/10.5281/zenodo.19366118

This is a clean, standalone implementation of the FINAL Analysis_05 protocol:

* 19 deterministic descriptors per frame;
* 44 descriptors per record (mean and standard deviation of the 19 frame
  descriptors plus six SSIM temporal summaries);
* a record-aware 70/15/15 split generated with ``random_state=42``;
* the same record membership at frame and record granularities;
* record-level leave-one-set-out (LOSO) evaluation;
* kNN: StandardScaler fitted on training data, k=7, distance weighting;
* Random Forest: 400 trees, square-root feature sampling, random_state=42.

The script defaults to both reported models. To run kNN only, use
``--models knn``. Feature extraction is the time-consuming part and is shared
by both models, so a single canonical script is preferable for publication.

Example
-------
python reproduce_classical_baselines.py

The default paths target the original Windows dataset location. All paths can
still be overridden with command-line arguments when required.

Expected dataset layout
-----------------------
FRS_dataset/
  labels.csv
  images/
    <filenames listed in labels.csv>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # Pillow fallback remains deterministic
    cv2 = None

from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
KNN_K = 7
RF_TREES = 400
SSIM_WIN = 11
SSIM_DATA_RANGE = 255
TARGET_COLS = ["force_p1_N", "force_p2_N", "force_p3_N"]
ID_COLS = ["set", "record_id"]
REQUIRED_COLS = [
    "filename",
    "set",
    "record_id",
    "frame_idx",
    "time_s",
    *TARGET_COLS,
]
DEFAULT_DATASET_DIR = Path(
    r"D:\others\python\EE701_Dataset_project\FRS_dataset"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "reproduction_runs"


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_gray_u8(path: Path) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image.astype(np.uint8)

    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def entropy_u8(image: np.ndarray) -> float:
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / (hist.sum() + 1e-12)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def speckle_contrast(image: np.ndarray) -> float:
    values = image.astype(np.float32)
    return float(np.std(values) / (np.mean(values) + 1e-12))


def psd_centroid(image: np.ndarray) -> float:
    values = image.astype(np.float32)
    values = values - values.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(values))
    power = np.abs(spectrum) ** 2
    height, width = power.shape
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    return float((power * radius).sum() / (power.sum() + 1e-12))


def grain_size_proxy_autocorr_halfmax(image: np.ndarray, max_r: int = 64) -> float:
    values = image.astype(np.float32)
    values = values - values.mean()
    spectrum = np.fft.fft2(values)
    autocorrelation = np.fft.ifft2(np.abs(spectrum) ** 2).real
    autocorrelation = np.fft.fftshift(autocorrelation)
    autocorrelation /= autocorrelation.max() + 1e-12

    height, width = autocorrelation.shape
    center_y, center_x = height // 2, width // 2
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    integer_radius = np.clip(radius.astype(int), 0, max_r)

    radial_profile = np.zeros(max_r + 1, dtype=np.float64)
    for r in range(max_r + 1):
        mask = integer_radius == r
        if mask.any():
            radial_profile[r] = autocorrelation[mask].mean()

    indices = np.where(radial_profile <= 0.5)[0]
    return float(indices[0]) if len(indices) else float(max_r)


def glcm_features(image: np.ndarray) -> dict[str, float]:
    levels = 32
    quantized = (
        image.astype(np.uint16) * (levels - 1) // 255
    ).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=[1],
        angles=[0],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    return {
        "glcm_contrast": float(graycoprops(matrix, "contrast")[0, 0]),
        "glcm_homogeneity": float(graycoprops(matrix, "homogeneity")[0, 0]),
        "glcm_energy": float(graycoprops(matrix, "energy")[0, 0]),
        "glcm_corr": float(graycoprops(matrix, "correlation")[0, 0]),
    }


def extract_frame_features(image: np.ndarray) -> dict[str, float]:
    values = image.astype(np.float32)
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / (hist.sum() + 1e-12)

    # Preserve the exact Analysis_05 column order. The order does not affect
    # standardized kNN distances, but it does affect the seeded RF feature
    # subsampling sequence and is therefore part of exact reproduction.
    features = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "iqr": float(p75 - p25),
        "entropy": entropy_u8(image),
        "energy": float((probabilities**2).sum()),
    }
    features.update(glcm_features(image))
    features.update(
        {
            "speckle_contrast": speckle_contrast(image),
            "ps_centroid": psd_centroid(image),
            "grain_halfmax_r": grain_size_proxy_autocorr_halfmax(image, max_r=64),
        }
    )
    return features


def compute_ssim_summaries(frames: list[np.ndarray], times: np.ndarray) -> dict[str, float]:
    reference = frames[0]
    values = np.asarray(
        [
            float(
                ssim(
                    reference,
                    image,
                    data_range=SSIM_DATA_RANGE,
                    win_size=SSIM_WIN,
                )
            )
            for image in frames
        ],
        dtype=np.float64,
    )
    shifted_times = np.asarray(times, dtype=np.float64)
    shifted_times = shifted_times - shifted_times.min()

    if np.std(shifted_times) < 1e-12:
        slope = 0.0
    else:
        slope = float(np.polyfit(shifted_times, values, 1)[0])

    duration = (
        float(shifted_times.max() - shifted_times.min())
        if len(shifted_times) > 1
        else 1.0
    )
    auc = float(np.trapz(values, shifted_times) / (duration + 1e-12))
    return {
        "ssim_mean": float(np.mean(values)),
        "ssim_std": float(np.std(values)),
        "ssim_min": float(np.min(values)),
        "ssim_tend": float(values[-1]),
        "ssim_slope": slope,
        "ssim_auc": auc,
    }


def validate_labels(df: pd.DataFrame, images_dir: Path, strict: bool) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"labels.csv is missing columns: {missing_columns}")

    df = df.copy()
    df["set"] = df["set"].astype(str)
    df["record_id"] = df["record_id"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["time_s"] = df["time_s"].astype(float)
    for column in TARGET_COLS:
        df[column] = df[column].astype(float)
    df = df.sort_values(["set", "record_id", "frame_idx"]).reset_index(drop=True)
    df["path"] = df["filename"].map(lambda name: str(images_dir / str(name)))

    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError("labels.csv contains missing required values.")
    if df["filename"].duplicated().any():
        raise ValueError("labels.csv contains duplicate filenames.")

    missing_images = [path for path in df["path"] if not Path(path).is_file()]
    if missing_images:
        preview = missing_images[:5]
        raise FileNotFoundError(
            f"{len(missing_images)} images are missing. First missing paths: {preview}"
        )

    if strict:
        grouped = df.groupby(ID_COLS, sort=True)
        sizes = grouped.size()
        if not (sizes == 20).all():
            raise ValueError("Strict validation failed: every record must contain 20 frames.")
        for key, group in grouped:
            expected_frames = list(range(20))
            if group["frame_idx"].tolist() != expected_frames:
                raise ValueError(f"Unexpected frame indices for record {key}.")
            if group[TARGET_COLS].drop_duplicates().shape[0] != 1:
                raise ValueError(f"Force labels vary within record {key}.")

    return df


def compute_feature_tables(
    df: pd.DataFrame,
    cache_dir: Path,
    reuse_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    frame_csv = cache_dir / "frame_features_all.csv"
    record_csv = cache_dir / "record_features_all.csv"
    if reuse_cache and frame_csv.exists() and record_csv.exists():
        print("[INFO] Reusing cached feature tables.")
        frame_df = pd.read_csv(frame_csv)
        record_df = pd.read_csv(record_csv)
    else:
        print(f"[INFO] Extracting frame features for {len(df):,} images...")
        start = time.time()
        rows: list[dict[str, object]] = []
        for index, row in df.iterrows():
            image = load_gray_u8(Path(row["path"]))
            features = extract_frame_features(image)
            rows.append(
                {
                    "filename": row["filename"],
                    "set": row["set"],
                    "record_id": int(row["record_id"]),
                    "frame_idx": int(row["frame_idx"]),
                    "time_s": float(row["time_s"]),
                    **features,
                    **{column: float(row[column]) for column in TARGET_COLS},
                }
            )
            if (index + 1) % 1000 == 0:
                print(f"  {index + 1:,}/{len(df):,}")
        frame_df = pd.DataFrame(rows)
        frame_df.to_csv(frame_csv, index=False)
        print(f"[INFO] Frame features completed in {time.time() - start:.1f} s.")

        frame_meta = ["filename", "set", "record_id", "frame_idx", "time_s", *TARGET_COLS]
        frame_feature_cols = [c for c in frame_df.columns if c not in frame_meta]

        print("[INFO] Aggregating record features and SSIM summaries...")
        start = time.time()
        record_rows: list[dict[str, object]] = []
        for (set_name, record_id), group in df.groupby(ID_COLS, sort=True):
            group = group.sort_values("frame_idx")
            frames = [load_gray_u8(Path(path)) for path in group["path"]]
            summaries = compute_ssim_summaries(
                frames, group["time_s"].to_numpy(dtype=np.float64)
            )
            frame_group = frame_df[
                (frame_df["set"] == set_name)
                & (frame_df["record_id"] == record_id)
            ].sort_values("frame_idx")

            aggregated: dict[str, float] = {}
            for column in frame_feature_cols:
                values = frame_group[column].to_numpy(dtype=np.float64)
                aggregated[f"{column}_mean"] = float(np.mean(values))
                aggregated[f"{column}_std"] = float(np.std(values))

            targets = group[TARGET_COLS].iloc[0]
            record_rows.append(
                {
                    "set": str(set_name),
                    "record_id": int(record_id),
                    **{column: float(targets[column]) for column in TARGET_COLS},
                    **summaries,
                    **aggregated,
                }
            )
        record_df = pd.DataFrame(record_rows)
        record_df.to_csv(record_csv, index=False)
        print(f"[INFO] Record features completed in {time.time() - start:.1f} s.")

    frame_meta = ["filename", "set", "record_id", "frame_idx", "time_s", *TARGET_COLS]
    frame_feature_cols = [c for c in frame_df.columns if c not in frame_meta]
    record_feature_cols = [c for c in record_df.columns if c not in [*ID_COLS, *TARGET_COLS]]
    if len(frame_feature_cols) != 19:
        raise ValueError(f"Expected 19 frame features; found {len(frame_feature_cols)}.")
    if len(record_feature_cols) != 44:
        raise ValueError(f"Expected 44 record features; found {len(record_feature_cols)}.")
    return frame_df, record_df, frame_feature_cols, record_feature_cols


def finite_matrix(values: np.ndarray, name: str, allow_legacy_imputation: bool) -> np.ndarray:
    matrix = values.astype(np.float64)
    matrix[np.isinf(matrix)] = np.nan
    count = int(np.isnan(matrix).sum())
    if count:
        if not allow_legacy_imputation:
            raise ValueError(
                f"{name} contains {count} nonfinite values. The final Analysis_05 code "
                "used full-matrix median imputation before splitting. Re-run with "
                "--allow-legacy-global-imputation only if exact legacy reproduction is "
                "required, or correct the feature-generation issue."
            )
        medians = np.nanmedian(matrix, axis=0)
        indices = np.where(np.isnan(matrix))
        matrix[indices] = np.take(medians, indices[1])
        print(f"[WARN] Applied legacy global-median imputation to {count} values in {name}.")
    return matrix.astype(np.float32)


def make_record_split(df: pd.DataFrame) -> dict[str, set[tuple[str, int]]]:
    records = df[ID_COLS].drop_duplicates().reset_index(drop=True)
    indices = np.arange(len(records))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )
    validation_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    train_indices, validation_indices = train_test_split(
        train_indices,
        test_size=validation_ratio,
        random_state=SEED,
        shuffle=True,
    )

    def keys(selected: Iterable[int]) -> set[tuple[str, int]]:
        return {
            (str(row["set"]), int(row["record_id"]))
            for _, row in records.iloc[list(selected)].iterrows()
        }

    return {
        "train": keys(train_indices),
        "val": keys(validation_indices),
        "test": keys(test_indices),
    }


def indices_for_keys(df: pd.DataFrame, keys: set[tuple[str, int]]) -> np.ndarray:
    mask = [
        (str(set_name), int(record_id)) in keys
        for set_name, record_id in zip(df["set"], df["record_id"])
    ]
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def save_splits(
    split_dir: Path,
    splits: dict[str, set[tuple[str, int]]],
    frame_df: pd.DataFrame,
    record_df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    for name, keys in splits.items():
        pd.DataFrame(sorted(keys), columns=ID_COLS).to_csv(
            split_dir / f"{name}_records.csv", index=False
        )
    frame_indices = {name: indices_for_keys(frame_df, keys) for name, keys in splits.items()}
    record_indices = {name: indices_for_keys(record_df, keys) for name, keys in splits.items()}
    np.savez(split_dir / "frame_indices_from_record_split.npz", **frame_indices)
    np.savez(split_dir / "record_indices.npz", **record_indices)
    return frame_indices, record_indices


def make_model(name: str):
    if name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        KNeighborsRegressor(n_neighbors=KNN_K, weights="distance")
                    ),
                ),
            ]
        )
    if name == "rf":
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=RF_TREES,
                random_state=SEED,
                n_jobs=-1,
                max_features="sqrt",
            )
        )
    raise ValueError(f"Unknown model: {name}")


def display_model_name(name: str) -> str:
    return {"knn": "kNN", "rf": "RandomForest"}[name]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    per_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    per_r2 = np.asarray(
        [r2_score(y_true[:, index], y_pred[:, index]) for index in range(3)],
        dtype=np.float64,
    )
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAE_P1": float(per_mae[0]),
        "MAE_P2": float(per_mae[1]),
        "MAE_P3": float(per_mae[2]),
        "RMSE_P1": float(per_rmse[0]),
        "RMSE_P2": float(per_rmse[1]),
        "RMSE_P3": float(per_rmse[2]),
        "R2_P1": float(per_r2[0]),
        "R2_P2": float(per_r2[1]),
        "R2_P3": float(per_r2[2]),
    }


def evaluate(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    model = make_model(model_key)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return compute_metrics(y_test, predictions)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_config(run_dir: Path, args: argparse.Namespace) -> None:
    versions = {
        name: version
        for name in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "scikit-image",
            "opencv-python",
            "Pillow",
        ]
        if (version := package_version(name)) is not None
    }
    # ``__file__`` is unavailable when this code is pasted into a Jupyter cell.
    script_name = globals().get("__file__")
    script_path = Path(script_name).resolve() if script_name else None
    config = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "labels_csv": str(args.labels_csv.resolve()),
        "images_dir": str(args.images_dir.resolve()),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "validation_size": VAL_SIZE,
        "models": list(args.models),
        "knn": {"n_neighbors": KNN_K, "weights": "distance", "scaled": True},
        "random_forest": {
            "n_estimators": RF_TREES,
            "max_features": "sqrt",
            "random_state": SEED,
            "scaled": False,
        },
        "ssim": {"win_size": SSIM_WIN, "data_range": SSIM_DATA_RANGE},
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "script_path": str(script_path) if script_path else None,
        "script_sha256": sha256(script_path) if script_path else None,
        "execution_context": "Python script" if script_path else "Jupyter notebook cell",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )


def compare_with_reference(
    produced: pd.DataFrame,
    reference_path: Path,
    output_path: Path,
    tolerance: float,
    strict: bool,
) -> None:
    reference = pd.read_csv(reference_path)
    metrics = [
        "MAE",
        "RMSE",
        "R2",
        "MAE_P1",
        "MAE_P2",
        "MAE_P3",
        "RMSE_P1",
        "RMSE_P2",
        "RMSE_P3",
        "R2_P1",
        "R2_P2",
        "R2_P3",
    ]
    keys = ["Block", "Model", "Fold"]
    for frame in [produced, reference]:
        if "Fold" not in frame.columns:
            frame["Fold"] = ""
        frame["Fold"] = frame["Fold"].fillna("")

    wanted_models = set(produced["Model"])
    reference = reference[reference["Model"].isin(wanted_models)].copy()
    merged = produced.merge(reference, on=keys, suffixes=("_new", "_reference"), how="outer", indicator=True)
    for metric in metrics:
        merged[f"abs_diff_{metric}"] = (
            merged[f"{metric}_new"] - merged[f"{metric}_reference"]
        ).abs()
    merged.to_csv(output_path, index=False)

    missing = merged["_merge"] != "both"
    maximum_difference = float(
        merged[[f"abs_diff_{metric}" for metric in metrics]].max().max()
    )
    passed = not missing.any() and maximum_difference <= tolerance
    print(
        f"[INFO] Reference comparison: max absolute metric difference "
        f"{maximum_difference:.3e}; tolerance {tolerance:.3e}; passed={passed}"
    )
    if strict and not passed:
        raise RuntimeError(
            f"Reference comparison failed. Inspect {output_path} for details."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset root containing labels.csv and images (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parent directory for timestamped output runs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional exact run directory. Use this with --reuse-cache to resume "
            "a previous feature-extraction run."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["knn", "rf"],
        default=["knn", "rf"],
        help="Reported models to run (default: knn rf).",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse feature CSVs in the current run cache if present.",
    )
    parser.add_argument(
        "--skip-strict-validation",
        action="store_true",
        help="Skip the checks for exactly 20 frames and constant labels per record.",
    )
    parser.add_argument(
        "--allow-legacy-global-imputation",
        action="store_true",
        help="Match Analysis_05 global-median handling if nonfinite features occur.",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=None,
        help="Optional original results_all.csv for automatic numeric comparison.",
    )
    parser.add_argument("--comparison-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--strict-reference",
        action="store_true",
        help="Exit with an error when the comparison exceeds the tolerance.",
    )
    # Notebook-paste mode: never read Jupyter's hidden ``sys.argv`` values.
    # All configured defaults above are used directly.
    args = parser.parse_args([])

    args.dataset_dir = args.dataset_dir.resolve()
    args.labels_csv = (
        args.labels_csv.resolve()
        if args.labels_csv is not None
        else args.dataset_dir / "labels.csv"
    )
    args.images_dir = (
        args.images_dir.resolve()
        if args.images_dir is not None
        else args.dataset_dir / "images"
    )
    return args


def main() -> None:
    args = parse_args()
    set_reproducibility(SEED)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else args.output_dir.resolve() / f"run_{timestamp}"
    )
    cache_dir = run_dir / "cache"
    split_dir = run_dir / "splits"
    results_dir = run_dir / "results"
    for directory in [cache_dir, split_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    save_config(run_dir, args)
    labels = validate_labels(
        pd.read_csv(args.labels_csv),
        args.images_dir,
        strict=not args.skip_strict_validation,
    )
    n_records = labels[ID_COLS].drop_duplicates().shape[0]
    print(
        f"[INFO] Validated {len(labels):,} frames, {n_records:,} records, "
        f"sets={sorted(labels['set'].unique())}."
    )

    frame_df, record_df, frame_features, record_features = compute_feature_tables(
        labels, cache_dir, reuse_cache=args.reuse_cache
    )
    splits = make_record_split(labels)
    frame_indices, record_indices = save_splits(
        split_dir, splits, frame_df, record_df
    )
    print(
        "[INFO] Record split sizes: "
        + ", ".join(f"{name}={len(keys)}" for name, keys in splits.items())
    )

    x_frame = finite_matrix(
        frame_df[frame_features].to_numpy(),
        "frame feature matrix",
        args.allow_legacy_global_imputation,
    )
    y_frame = frame_df[TARGET_COLS].to_numpy(dtype=np.float32)
    x_record = finite_matrix(
        record_df[record_features].to_numpy(),
        "record feature matrix",
        args.allow_legacy_global_imputation,
    )
    y_record = record_df[TARGET_COLS].to_numpy(dtype=np.float32)

    rows: list[dict[str, object]] = []
    for model_key in args.models:
        metrics = evaluate(
            model_key,
            x_frame[frame_indices["train"]],
            y_frame[frame_indices["train"]],
            x_frame[frame_indices["test"]],
            y_frame[frame_indices["test"]],
        )
        rows.append(
            {
                "Block": "FrameLevel_RecordSplit_TEST",
                "Model": display_model_name(model_key),
                "Granularity": "Frame",
                "Protocol": "RecordSplit",
                "N_train": len(frame_indices["train"]),
                "N_test": len(frame_indices["test"]),
                **metrics,
                "Fold": "",
            }
        )
        print(
            f"[OK] Frame RecordSplit | {display_model_name(model_key)} | "
            f"MAE={metrics['MAE']:.6f} R2={metrics['R2']:.6f}"
        )

        metrics = evaluate(
            model_key,
            x_record[record_indices["train"]],
            y_record[record_indices["train"]],
            x_record[record_indices["test"]],
            y_record[record_indices["test"]],
        )
        rows.append(
            {
                "Block": "RecordLevel_RecordSplit_TEST",
                "Model": display_model_name(model_key),
                "Granularity": "Record",
                "Protocol": "RecordSplit",
                "N_train": len(record_indices["train"]),
                "N_test": len(record_indices["test"]),
                **metrics,
                "Fold": "",
            }
        )
        print(
            f"[OK] Record RecordSplit | {display_model_name(model_key)} | "
            f"MAE={metrics['MAE']:.6f} R2={metrics['R2']:.6f}"
        )

    print("[INFO] Running record-level leave-one-set-out evaluation...")
    set_values = record_df["set"].astype(str).to_numpy()
    for held_out_set in sorted(record_df["set"].astype(str).unique()):
        train_indices = np.flatnonzero(set_values != held_out_set)
        test_indices = np.flatnonzero(set_values == held_out_set)
        for model_key in args.models:
            metrics = evaluate(
                model_key,
                x_record[train_indices],
                y_record[train_indices],
                x_record[test_indices],
                y_record[test_indices],
            )
            rows.append(
                {
                    "Block": "RecordLevel_LOSO",
                    "Model": display_model_name(model_key),
                    "Granularity": "Record",
                    "Protocol": "LOSO",
                    "N_train": len(train_indices),
                    "N_test": len(test_indices),
                    **metrics,
                    "Fold": f"LOSO_{held_out_set}",
                }
            )
            print(
                f"[OK] LOSO_{held_out_set} | {display_model_name(model_key)} | "
                f"MAE={metrics['MAE']:.6f} R2={metrics['R2']:.6f}"
            )

    column_order = [
        "Block",
        "Model",
        "Granularity",
        "Protocol",
        "N_train",
        "N_test",
        "MAE",
        "RMSE",
        "R2",
        "MAE_P1",
        "MAE_P2",
        "MAE_P3",
        "RMSE_P1",
        "RMSE_P2",
        "RMSE_P3",
        "R2_P1",
        "R2_P2",
        "R2_P3",
        "Fold",
    ]
    results = pd.DataFrame(rows)[column_order]
    results_path = results_dir / "results_all.csv"
    results.to_csv(results_path, index=False)

    loso_summary = (
        results[results["Protocol"] == "LOSO"]
        .groupby("Model")[["MAE", "RMSE", "R2"]]
        .mean()
        .reset_index()
    )
    loso_summary.to_csv(results_dir / "loso_mean_metrics.csv", index=False)

    if args.reference_results is not None:
        compare_with_reference(
            results,
            args.reference_results.resolve(),
            results_dir / "reference_comparison.csv",
            tolerance=args.comparison_tolerance,
            strict=args.strict_reference,
        )

    print(f"[DONE] Reproduction outputs: {run_dir}")
    print(f"[DONE] Main results: {results_path}")


if __name__ == "__main__":
    main()





