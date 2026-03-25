"""
Main entry point: orchestrates zero-shot and few-shot experiments.

Usage:
    python run_all.py                          # Run full pipeline
    python run_all.py --mode zero_shot         # Zero-shot only
    python run_all.py --mode few_shot          # Few-shot only
    python run_all.py --max-samples 200        # Quick test with 200 samples
    python run_all.py --shots 4 8              # Few-shot with specific k values
"""

import argparse
import sys
import os
import joblib

import numpy as np
import torch
from loguru import logger

# Add parent directory to path so we can import open_clip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (
    MODELS,
    CATEGORIES,
    FEW_SHOT_K_VALUES,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    SEED,
    DEVICE,
)
from dataset import prepare_dataset
from zero_shot import run_zero_shot
from few_shot import run_few_shot
from wise_ft import run_wise_ft
from evaluate import compute_metrics, print_metrics, save_results
from visualize import (
    plot_confusion_matrix,
    plot_zero_shot_comparison,
    plot_few_shot_curves,
    plot_zero_vs_few_shot,
    plot_label_distribution,
    plot_wise_ft,
    generate_summary_table,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_zero_shot_experiments(models, dataset):
    """Run zero-shot classification with all models."""
    all_results = {}

    for model_cfg in models:
        name = model_cfg["name"]
        pretrained = model_cfg["pretrained"]

        preds, labels, probs, classifier = run_zero_shot(name, pretrained, dataset)

        # Save Zero-Shot textual classification head
        model_filename = name.replace("/", "_").replace("-", "_")
        torch.save(
            classifier,
            os.path.join(SAVED_MODELS_DIR, f"{model_filename}_zeroshot_head.pt"),
        )
        logger.info(
            f"Saved Zero-Shot head to {SAVED_MODELS_DIR}/{model_filename}_zeroshot_head.pt"
        )
        metrics = compute_metrics(preds, labels, prefix="zs")
        print_metrics(metrics, title=f"Zero-Shot: {name}")

        # Save confusion matrix plot
        cm = np.array(metrics["zs_confusion_matrix"])
        plot_confusion_matrix(
            cm,
            title=f"Zero-Shot Confusion Matrix: {name}",
            filename=f"cm_zero_shot_{name.replace('-', '_')}.png",
        )

        all_results[name] = {
            "accuracy": metrics["zs_accuracy"],
            "f1_weighted": metrics["zs_f1_weighted"],
            "f1_macro": metrics["zs_f1_macro"],
            "precision_weighted": metrics["zs_precision_weighted"],
            "recall_weighted": metrics["zs_recall_weighted"],
            "confusion_matrix": metrics["zs_confusion_matrix"],
            "per_class": metrics["zs_per_class"],
            "num_samples": metrics["zs_num_samples"],
        }

    # Save combined results
    save_results(all_results, "zero_shot_results.json")

    # Plot comparison
    if len(all_results) > 1:
        plot_zero_shot_comparison(all_results)

    return all_results


def run_few_shot_experiments(models, dataset, k_values):
    """Run few-shot classification with all models and k values."""
    all_results = {}

    for model_cfg in models:
        name = model_cfg["name"]
        pretrained = model_cfg["pretrained"]

        k_results = run_few_shot(name, pretrained, dataset, k_values)

        model_results = {}
        best_overall_acc = 0.0
        best_overall_model = None

        for k, res in k_results.items():
            metrics = compute_metrics(res["preds"], res["labels"], prefix=f"fs_{k}")
            print_metrics(metrics, title=f"Few-Shot (k={k}): {name}")

            # Keep track of the absolute best few-shot model across all k variants
            if res["accuracy_best"] > best_overall_acc:
                best_overall_acc = res["accuracy_best"]
                best_overall_model = res.get("best_model", None)

            # Save confusion matrix
            cm = np.array(metrics[f"fs_{k}_confusion_matrix"])
            plot_confusion_matrix(
                cm,
                title=f"Few-Shot (k={k}) Confusion Matrix: {name}",
                filename=f"cm_few_shot_{name.replace('-', '_')}_k{k}.png",
            )

            model_results[k] = {
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "accuracy": metrics[f"fs_{k}_accuracy"],
                "f1_weighted": metrics[f"fs_{k}_f1_weighted"],
                "f1_macro": metrics[f"fs_{k}_f1_macro"],
                "confusion_matrix": metrics[f"fs_{k}_confusion_matrix"],
                "num_samples": metrics[f"fs_{k}_num_samples"],
            }

        # Save the best Linear Probe model for this architecture to disk
        if best_overall_model is not None:
            model_filename = name.replace("/", "_").replace("-", "_")
            joblib.dump(
                best_overall_model,
                os.path.join(SAVED_MODELS_DIR, f"{model_filename}_best_fewshot.joblib"),
            )
            logger.info(
                f"Saved best Few-Shot model to {SAVED_MODELS_DIR}/{model_filename}_best_fewshot.joblib"
            )

        all_results[name] = model_results

    # Save combined results
    save_results(all_results, "few_shot_results.json")

    # Plot few-shot curves
    if all_results:
        plot_few_shot_curves(all_results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Zero-Shot & Few-Shot Classification with OpenCLIP"
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot", "wise_ft", "all"],
        default="all",
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size for quick testing",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help="Few-shot k values (default: from config)",
    )
    parser.add_argument(
        "--models",
        type=int,
        nargs="+",
        default=None,
        help="Model indices from config to use (e.g., 0 1)",
    )
    args = parser.parse_args()

    set_seed(SEED)

    k_values = args.shots if args.shots else FEW_SHOT_K_VALUES
    models = MODELS
    if args.models:
        models = [MODELS[i] for i in args.models if i < len(MODELS)]

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Models: {[m['name'] for m in models]}")
    logger.info(f"Categories: {CATEGORIES}")
    if args.mode in ("few_shot", "all"):
        logger.info(f"Few-shot k values: {k_values}")

    # ─── Prepare dataset ─────────────────────────────────────────────────
    # Use a SEPARATE labeler model (ViT-L-14) for pseudo-label assignment
    # so evaluation models don't "grade their own test"
    logger.info("STEP 1: Preparing dataset")
    logger.info("Loading CIFAR-100 from HuggingFace...")
    dataset, label_dist = prepare_dataset(
        max_samples=args.max_samples,
    )
    logger.info(f"Dataset size: {len(dataset)}")
    plot_label_distribution(label_dist)

    # ─── Zero-shot experiments ───────────────────────────────────────────
    zero_results = {}
    if args.mode in ("zero_shot", "all"):
        logger.info("STEP 2: Zero-Shot Classification")
        zero_results = run_zero_shot_experiments(models, dataset)

    # ─── Few-shot experiments ────────────────────────────────────────────
    few_results = {}
    if args.mode in ("few_shot", "all"):
        logger.info("STEP 3: Few-Shot Classification")
        few_results = run_few_shot_experiments(models, dataset, k_values)

    # ─── WiSE-FT experiments ─────────────────────────────────────────────
    wise_ft_results = {}
    if args.mode in ("wise_ft", "all"):
        logger.info("STEP 4: WiSE-FT Classification")
        for model_cfg in models:
            name = model_cfg["name"]
            pretrained = model_cfg["pretrained"]

            # Using the largest k value for robust fine-tuning
            best_k = k_values[-1] if k_values else 50
            wise_result, best_model_sd = run_wise_ft(
                name, pretrained, dataset, k=best_k, epochs=3
            )

            # Save the optimal alpha-mixed WiSE-FT state dictated weights
            model_filename = name.replace("/", "_").replace("-", "_")
            torch.save(
                best_model_sd,
                os.path.join(SAVED_MODELS_DIR, f"{model_filename}_best_wise_ft.pt"),
            )
            logger.info(
                f"Saved best WiSE-FT model to {SAVED_MODELS_DIR}/{model_filename}_best_wise_ft.pt"
            )

            wise_ft_results[name] = wise_result
        save_results(wise_ft_results, "wise_ft_results.json")

    # ─── Summary ─────────────────────────────────────────────────────────
    if zero_results and few_results:
        logger.info("STEP 5: Summary & Visualization")
        generate_summary_table(zero_results, few_results)
        plot_zero_vs_few_shot(zero_results, few_results)

    if wise_ft_results:
        plot_wise_ft(wise_ft_results)

    # Save overall summary
    summary = {
        "device": DEVICE,
        "categories": CATEGORIES,
        "models": [m["name"] for m in models],
        "dataset_size": len(dataset),
        "label_distribution": label_dist,
    }
    if zero_results:
        summary["zero_shot"] = {
            m: {"accuracy": r["accuracy"], "f1_weighted": r["f1_weighted"]}
            for m, r in zero_results.items()
        }
    if few_results:
        summary["few_shot"] = {
            m: {
                str(k): {
                    "accuracy_mean": v["accuracy_mean"],
                    "accuracy_std": v["accuracy_std"],
                }
                for k, v in kv.items()
            }
            for m, kv in few_results.items()
        }
    if wise_ft_results:
        summary["wise_ft"] = {
            m: {
                str(alpha): {"accuracy": v["accuracy"], "f1_weighted": v["f1_weighted"]}
                for alpha, v in res.items()
            }
            for m, res in wise_ft_results.items()
        }
    save_results(summary, "experiment_summary.json")

    logger.success("\n✅ All experiments completed!")
    logger.info(f"   Results saved in: {RESULTS_DIR}")
    logger.info(
        f"   Plots saved in:   {os.path.join(os.path.dirname(__file__), 'plots')}"
    )


if __name__ == "__main__":
    main()
