#!/usr/bin/env python3
"""Upload BiTempQA dataset to HuggingFace.

Usage:
    # First time: login
    huggingface-cli login

    # Upload:
    python upload_to_huggingface.py
"""

from huggingface_hub import HfApi, create_repo
import os

REPO_ID = "heihei/BiTempQA"  # Change to your HuggingFace username
DATASET_DIR = os.path.join(os.path.dirname(__file__), "huggingface_dataset")

def main():
    api = HfApi()

    # Create repo (skip if exists)
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"Repository {REPO_ID} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload dataset card
    api.upload_file(
        path_or_fileobj=os.path.join(DATASET_DIR, "dataset_card.md"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("Uploaded README.md")

    # Upload data files
    for filename in ["train.json", "test.json", "dev.json"]:
        filepath = os.path.join(DATASET_DIR, filename)
        if os.path.exists(filepath):
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print(f"Uploaded {filename}")

    # Upload scenario templates
    templates_dir = os.path.join(DATASET_DIR, "scenario_templates")
    if os.path.exists(templates_dir):
        for f in os.listdir(templates_dir):
            if f.endswith(".md"):
                api.upload_file(
                    path_or_fileobj=os.path.join(templates_dir, f),
                    path_in_repo=f"scenario_templates/{f}",
                    repo_id=REPO_ID,
                    repo_type="dataset",
                )
                print(f"Uploaded scenario_templates/{f}")

    # Upload seed prompts
    prompts_dir = os.path.join(DATASET_DIR, "seed_prompts")
    if os.path.exists(prompts_dir):
        for f in os.listdir(prompts_dir):
            api.upload_file(
                path_or_fileobj=os.path.join(prompts_dir, f),
                path_in_repo=f"seed_prompts/{f}",
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print(f"Uploaded seed_prompts/{f}")

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
