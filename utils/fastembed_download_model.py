#!/usr/bin/env python3
"""
List all models supported by FastEmbed's TextEmbedding and download selected models.

Usage:
    python utils/fastembed_download_model.py [--cache-dir PATH]

Options:
    --cache-dir  Directory where models are cached (default: ./models/fastembed_cache)
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

from fastembed import TextEmbedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="List FastEmbed models and download selected models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    python utils/fastembed_download_model.py
    python utils/fastembed_download_model.py --cache-dir ./my_cache
""",
    )
    parser.add_argument(
        "--cache-dir",
        default="./models/fastembed_cache",
        help="Directory where models are cached (default: ./models/fastembed_cache)",
    )
    return parser.parse_args()


def print_supported_models(models):
    print(f"FastEmbed supported models ({len(models)}):")
    print("=" * 90)
    print(f"{'#':<4} {'Model':<45} {'Dim':<6} {'Size (GB)':<10} Description")
    print("-" * 90)
    for i, m in enumerate(models, 1):
        model_name = m.get("model", "")
        dim = m.get("dim", "")
        size = m.get("size_in_GB", "")
        desc = m.get("description", "")
        print(f"{i:<4} {model_name:<45} {str(dim):<6} {str(size):<10} {desc}")


def get_selection(models):
    prompt = "Enter model number to download (or 'q' to quit): "
    while True:
        choice = input(prompt).strip()
        if choice.lower() == "q":
            print("Goodbye.")
            sys.exit(0)
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                return idx - 1
            print(f"Invalid selection. Please enter a number between 1 and {len(models)}, or 'q'.")
        except ValueError:
            print(f"Invalid selection. Please enter a number between 1 and {len(models)}, or 'q'.")


def confirm_download(model_name, cache_dir):
    print()
    print(f"You are about to download: {model_name}")
    print(f"Download path: {cache_dir}")
    cache_path = Path(cache_dir)
    if cache_path.exists() and any(cache_path.iterdir()):
        print("Warning: a model is already present in this directory.")
        print("It will be deleted and replaced by the selected model.")
    print()
    while True:
        confirm = input("Proceed? [y/N]: ").strip()
        if confirm.lower() == "y":
            return True
        print("Download cancelled.")
        return False


def download_model(model_name, cache_dir):
    cache_path = Path(cache_dir).resolve()
    if cache_path.exists() and any(cache_path.iterdir()):
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("Downloading model (this may take a few minutes on first run)...")
    start = time.perf_counter()
    model = TextEmbedding(model_name=model_name, cache_dir=str(cache_path))
    elapsed = time.perf_counter() - start
    print(f"Model downloaded successfully in {elapsed:.1f}s")
    print(f"Model cached at: {cache_path}")


def main():
    args = parse_args()
    models = list(TextEmbedding.list_supported_models())

    print_supported_models(models)
    print()
    idx = get_selection(models)
    model_name = models[idx].get("model", "")

    if not confirm_download(model_name, args.cache_dir):
        return

    download_model(model_name, args.cache_dir)
    print()
    print("Download complete. Goodbye.")


if __name__ == "__main__":
    main()
