#!/usr/bin/env python3
"""
Compare similarity between two figure files (PNG/JPG/TIF/PDF).

Outputs:
1) Numeric similarity metrics (SSIM, PSNR, MSE, MAE, perceptual hash distance)
2) Diff images:
   - diff_ssim.png: SSIM difference map (brighter = more different)
   - diff_abs.png: absolute pixel difference heatmap (grayscale)
   - overlay.png: alpha overlay for quick visual inspection

Usage:
  python compare_figures.py path/to/fig1.png path/to/fig2.png --outdir out
  python compare_figures.py fig1.pdf fig2.pdf --outdir out --pdf-page 0
  python compare_figures.py fig1.png fig2.png --align orb
  python compare_figures.py fig1.png fig2.png --compare-both --outdir out
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image

try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV is required. Install with: pip install opencv-python", file=sys.stderr)
    raise

try:
    from skimage.metrics import structural_similarity as ssim
except Exception as e:
    print("ERROR: scikit-image is required. Install with: pip install scikit-image", file=sys.stderr)
    raise

try:
    import imagehash
except Exception:
    imagehash = None


DEFAULT_CLASSIFICATION_THRESHOLDS = {
    "same": 0.995,
    "highly_similar": 0.950,
    "partially_similar": 0.800,
}


def load_image_any(path: str, pdf_page: int = 0) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
        img = Image.open(p).convert("RGB")
        return img

    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except Exception:
            raise RuntimeError("PDF input requires PyMuPDF. Install with: pip install pymupdf")

        doc = fitz.open(str(p))
        if pdf_page < 0 or pdf_page >= len(doc):
            raise ValueError(f"pdf_page {pdf_page} out of range (0..{len(doc)-1})")

        page = doc.load_page(pdf_page)
        pix = page.get_pixmap(alpha=False)  # RGB
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    raise ValueError(f"Unsupported file type: {ext}")


def pil_to_cv_rgb(img: Image.Image) -> np.ndarray:
    arr = np.array(img)  # RGB
    return arr


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def match_sizes(img1: np.ndarray, img2: np.ndarray, mode: str = "resize") -> tuple[np.ndarray, np.ndarray]:
    if img1.shape[:2] == img2.shape[:2]:
        return img1, img2

    if mode == "resize":
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1r = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        img2r = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        return img1r, img2r

    if mode == "pad":
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])

        def pad_to(a: np.ndarray) -> np.ndarray:
            top = 0
            left = 0
            bottom = h - a.shape[0]
            right = w - a.shape[1]
            return cv2.copyMakeBorder(a, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        return pad_to(img1), pad_to(img2)

    raise ValueError("mode must be 'resize' or 'pad'")


def align_orb(src_rgb: np.ndarray, dst_rgb: np.ndarray) -> np.ndarray:
    """
    Warp src onto dst using ORB keypoints + homography.
    Returns aligned src (same size as dst). If alignment fails, returns resized src.
    """
    src = src_rgb.copy()
    dst = dst_rgb.copy()
    src_g = to_gray(src)
    dst_g = to_gray(dst)

    orb_create = getattr(cv2, "ORB_create", None)
    if orb_create is None:
        return cv2.resize(src, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_AREA)
    orb = orb_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(src_g, None)
    kp2, des2 = orb.detectAndCompute(dst_g, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return cv2.resize(src, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_AREA)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 12:
        return cv2.resize(src, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_AREA)

    matches = sorted(matches, key=lambda m: m.distance)[:500]
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        return cv2.resize(src, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_AREA)

    aligned = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return aligned


def compute_metrics(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> dict:
    g1 = to_gray(img1_rgb)
    g2 = to_gray(img2_rgb)

    g1f = g1.astype(np.float32)
    g2f = g2.astype(np.float32)

    mse = float(np.mean((g1f - g2f) ** 2))
    mae = float(np.mean(np.abs(g1f - g2f)))

    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = 20.0 * float(np.log10(255.0 / np.sqrt(mse)))

    ssim_result = ssim(g1, g2, data_range=255, full=True)
    if not isinstance(ssim_result, tuple) or len(ssim_result) < 2:
        raise RuntimeError("Failed to compute SSIM map")
    ssim_score, ssim_map = ssim_result[0], ssim_result[1]

    phash_dist = None
    if imagehash is not None:
        h1 = imagehash.phash(Image.fromarray(img1_rgb))
        h2 = imagehash.phash(Image.fromarray(img2_rgb))
        phash_dist = int(h1 - h2)

    return {
        "ssim": float(ssim_score),
        "psnr": psnr,
        "mse": mse,
        "mae": mae,
        "phash_distance": phash_dist,
        "ssim_map": ssim_map,
    }


def save_outputs(outdir: Path, img1_rgb: np.ndarray, img2_rgb: np.ndarray, ssim_map: np.ndarray) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    g1 = to_gray(img1_rgb)
    g2 = to_gray(img2_rgb)

    absdiff = cv2.absdiff(g1, g2)  # 0..255

    ssim_diff = (1.0 - ssim_map)  # 0..1
    ssim_diff_u8 = np.clip(ssim_diff * 255.0, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(img1_rgb, 0.5, img2_rgb, 0.5, 0)

    Image.fromarray(absdiff).save(outdir / "diff_abs.png")
    Image.fromarray(ssim_diff_u8).save(outdir / "diff_ssim.png")
    Image.fromarray(overlay).save(outdir / "overlay.png")
    Image.fromarray(img1_rgb).save(outdir / "img1_used.png")
    Image.fromarray(img2_rgb).save(outdir / "img2_used.png")


def compare_figures(
    figure1: str | Path | Image.Image | np.ndarray,
    figure2: str | Path | Image.Image | np.ndarray,
    *,
    outdir: str | Path = "figure_compare_out",
    pdf_page: int = 0,
    size_mode: str = "resize",
    align: str = "none",
    include_phash: bool = True,
    thresholds: dict[str, float] | None = None,
    score_from: str = "ssim",
) -> dict[str, Any]:
    """
    Compare two figures, compute complementary similarity metrics, and return an
    interpretable conclusion.

    The function accepts file paths (including PDF) or in-memory images, standardizes
    both inputs to RGB arrays, harmonizes resolution, optionally aligns geometry,
    computes metrics, writes diagnostic images, and returns a decision payload.

    Parameters
    ----------
    figure1, figure2:
        Figure inputs. Each can be:
        - filesystem path (`str`/`Path`) to image/PDF,
        - `PIL.Image.Image`,
        - `numpy.ndarray` (grayscale/RGB/RGBA).
    outdir:
        Directory where diagnostics are saved:
        `diff_abs.png`, `diff_ssim.png`, `overlay.png`, `img1_used.png`, `img2_used.png`.
    pdf_page:
        Page index (0-based) when input is a PDF path.
    size_mode:
        How to standardize different resolutions: `"resize"` or `"pad"`.
    align:
        Optional geometric alignment strategy: `"none"` or `"orb"`.
    include_phash:
        If `True`, compute perceptual hash distance when `imagehash` is installed.
    thresholds:
        Classification thresholds based on SSIM, with keys:
        `same`, `highly_similar`, `partially_similar`.
        Defaults: 0.995, 0.95, 0.80.
    score_from:
        Metric used to generate a 0-100 similarity score. Default `"ssim"`.
        Currently supports `"ssim"`.

    Returns
    -------
    dict
        Result dictionary containing:
        - `metrics`: SSIM, PSNR, MSE, MAE, optional pHash distance,
        - `similarity_score`: percentage in [0, 100],
        - `classification`: one of
          `"same"`, `"highly similar"`, `"partially similar"`, `"not similar"`,
        - `thresholds_used`,
        - `diagnostics_dir`,
        - `conclusion` (human-readable one-line summary).

    Example
    -------
    >>> result = compare_figures("fig1.png", "fig2.png", outdir="out")
    >>> print(
    ...     f"Figures are {result['similarity_score']:.1f}% similar "
    ...     f"(SSIM = {result['metrics']['ssim']:.4f}); "
    ...     f"classification: {result['classification']}"
    ... )
    """

    if size_mode not in {"resize", "pad"}:
        raise ValueError("size_mode must be 'resize' or 'pad'")
    if align not in {"none", "orb"}:
        raise ValueError("align must be 'none' or 'orb'")

    thresholds_used = dict(DEFAULT_CLASSIFICATION_THRESHOLDS)
    if thresholds is not None:
        thresholds_used.update(thresholds)

    required_keys = {"same", "highly_similar", "partially_similar"}
    if not required_keys.issubset(thresholds_used.keys()):
        raise ValueError("thresholds must include keys: same, highly_similar, partially_similar")

    if not (
        0.0 <= thresholds_used["partially_similar"] <= thresholds_used["highly_similar"] <= thresholds_used["same"] <= 1.0
    ):
        raise ValueError(
            "thresholds must satisfy 0 <= partially_similar <= highly_similar <= same <= 1"
        )

    def to_rgb_array(inp: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(inp, (str, Path)):
            pil_img = load_image_any(str(inp), pdf_page=pdf_page)
            return pil_to_cv_rgb(pil_img)

        if isinstance(inp, Image.Image):
            return pil_to_cv_rgb(inp.convert("RGB"))

        if isinstance(inp, np.ndarray):
            arr = inp
            if arr.ndim == 2:
                return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)

            if arr.ndim == 3 and arr.shape[2] == 4:
                return arr[:, :, :3].astype(np.uint8)

            if arr.ndim == 3 and arr.shape[2] == 3:
                return arr.astype(np.uint8)

            raise ValueError("Unsupported numpy array shape for image input")

        raise TypeError("figure input must be a path, PIL.Image.Image, or numpy.ndarray")

    img1_rgb = to_rgb_array(figure1)
    img2_rgb = to_rgb_array(figure2)

    img1_rgb, img2_rgb = match_sizes(img1_rgb, img2_rgb, mode=size_mode)

    if align == "orb":
        img1_aligned = align_orb(img1_rgb, img2_rgb)
        img1_rgb, img2_rgb = match_sizes(img1_aligned, img2_rgb, mode="resize")

    metrics = compute_metrics(img1_rgb, img2_rgb)
    if not include_phash:
        metrics["phash_distance"] = None

    if score_from != "ssim":
        raise ValueError("score_from currently supports only 'ssim'")

    similarity_score = float(np.clip(metrics["ssim"] * 100.0, 0.0, 100.0))

    ssim_value = metrics["ssim"]
    if ssim_value >= thresholds_used["same"]:
        classification = "same"
    elif ssim_value >= thresholds_used["highly_similar"]:
        classification = "highly similar"
    elif ssim_value >= thresholds_used["partially_similar"]:
        classification = "partially similar"
    else:
        classification = "not similar"

    output_dir = Path(outdir)
    save_outputs(output_dir, img1_rgb, img2_rgb, metrics["ssim_map"])

    conclusion = (
        f"Figures are {similarity_score:.1f}% similar "
        f"(SSIM = {metrics['ssim']:.4f}); classification: {classification}"
    )

    return {
        "metrics": {
            "ssim": float(metrics["ssim"]),
            "psnr": float(metrics["psnr"]),
            "mse": float(metrics["mse"]),
            "mae": float(metrics["mae"]),
            "phash_distance": metrics["phash_distance"],
        },
        "similarity_score": similarity_score,
        "classification": classification,
        "thresholds_used": {
            "same": float(thresholds_used["same"]),
            "highly_similar": float(thresholds_used["highly_similar"]),
            "partially_similar": float(thresholds_used["partially_similar"]),
        },
        "diagnostics_dir": str(output_dir.resolve()),
        "conclusion": conclusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig1", help="Path to first figure (png/jpg/tif/pdf)")
    ap.add_argument("fig2", help="Path to second figure (png/jpg/tif/pdf)")
    ap.add_argument("--outdir", default="figure_compare_out", help="Output directory")
    ap.add_argument("--pdf-page", type=int, default=0, help="PDF page index (0-based)")
    ap.add_argument("--size-mode", choices=["resize", "pad"], default="resize", help="How to handle different sizes")
    ap.add_argument("--align", choices=["none", "orb"], default="none", help="Optional geometric alignment")
    ap.add_argument(
        "--compare-both",
        action="store_true",
        help="Run both alignment modes (none and orb) and print a side-by-side summary",
    )
    args = ap.parse_args()

    def print_result(result: dict[str, Any]) -> None:
        metrics = result["metrics"]
        print("Similarity metrics")
        print(f"SSIM: {metrics['ssim']:.6f}  (1.0 means identical structure)")
        print(f"PSNR: {metrics['psnr']:.3f} dB  (higher means more similar)")
        print(f"MSE : {metrics['mse']:.3f}")
        print(f"MAE : {metrics['mae']:.3f}")
        if metrics["phash_distance"] is None:
            print("pHash distance: not computed (install imagehash: pip install ImageHash)")
        else:
            print(f"pHash distance: {metrics['phash_distance']}  (0 means identical perceptual hash)")
        print()
        print(result["conclusion"])
        print(f"Saved outputs to: {result['diagnostics_dir']}")
        print("Files: diff_abs.png, diff_ssim.png, overlay.png, img1_used.png, img2_used.png")

    if args.compare_both:
        base_out = Path(args.outdir)
        result_none = compare_figures(
            args.fig1,
            args.fig2,
            outdir=base_out / "none",
            pdf_page=args.pdf_page,
            size_mode=args.size_mode,
            align="none",
            include_phash=True,
        )
        result_orb = compare_figures(
            args.fig1,
            args.fig2,
            outdir=base_out / "orb",
            pdf_page=args.pdf_page,
            size_mode=args.size_mode,
            align="orb",
            include_phash=True,
        )

        print("Comparison summary (none vs orb)")
        print(f"{'Mode':<8} {'SSIM':>9} {'Score%':>9} {'PSNR(dB)':>10} {'MSE':>12} {'MAE':>10} {'Class':>18}")
        print(
            f"{'none':<8} "
            f"{result_none['metrics']['ssim']:>9.6f} "
            f"{result_none['similarity_score']:>9.1f} "
            f"{result_none['metrics']['psnr']:>10.3f} "
            f"{result_none['metrics']['mse']:>12.3f} "
            f"{result_none['metrics']['mae']:>10.3f} "
            f"{result_none['classification']:>18}"
        )
        print(
            f"{'orb':<8} "
            f"{result_orb['metrics']['ssim']:>9.6f} "
            f"{result_orb['similarity_score']:>9.1f} "
            f"{result_orb['metrics']['psnr']:>10.3f} "
            f"{result_orb['metrics']['mse']:>12.3f} "
            f"{result_orb['metrics']['mae']:>10.3f} "
            f"{result_orb['classification']:>18}"
        )
        print()
        print(f"none diagnostics: {result_none['diagnostics_dir']}")
        print(f"orb diagnostics : {result_orb['diagnostics_dir']}")
        return

    result = compare_figures(
        args.fig1,
        args.fig2,
        outdir=args.outdir,
        pdf_page=args.pdf_page,
        size_mode=args.size_mode,
        align=args.align,
        include_phash=True,
    )
    print_result(result)


if __name__ == "__main__":
    main()