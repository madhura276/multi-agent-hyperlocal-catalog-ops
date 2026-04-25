from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset # type: ignore
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer # type: ignore


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SPACE_REPO_ID = "MadhuraMadhu/multi-agent-hyperlocal-catalog-ops"
OUTPUT_DIR = "outputs/oversight-sft"
SAVE_REPO_ID = os.environ.get("SAVE_REPO_ID", SPACE_REPO_ID)
SAVE_REPO_TYPE = os.environ.get("SAVE_REPO_TYPE", "space")
SAVE_PATH_IN_REPO = os.environ.get("SAVE_PATH_IN_REPO", "trained_models/oversight-sft-final")


def resolve_data_path() -> str:
    script_path = Path(__file__).resolve()
    candidate_roots = [script_path.parent, *script_path.parents]

    for root in candidate_roots:
        local_path = root / "data" / "oversight_sft.jsonl"
        if local_path.exists():
            return str(local_path)

    downloaded_path = hf_hub_download(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        filename="data/oversight_sft.jsonl",
    )
    return downloaded_path


def format_example(example: dict) -> dict:
    return {"text": f"{example['prompt']}\n{example['completion']}"}


def main() -> None:
    data_path = resolve_data_path()
    print(f"Loading SFT data from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(format_example)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            logging_steps=1,
            save_strategy="no",
            max_length=1024,
            report_to=[],
            completion_only_loss=False,
            dataset_text_field="text",
            bf16=True,
            fp16=False,
        ),
    )
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}-final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")

    print(f"Saved trained model to {OUTPUT_DIR}-final")
    api = HfApi()
    api.upload_folder(
        repo_id=SAVE_REPO_ID,
        repo_type=SAVE_REPO_TYPE,
        folder_path=f"{OUTPUT_DIR}-final",
        path_in_repo=SAVE_PATH_IN_REPO,
        commit_message="Add trained oversight agent model",
    )
    print(f"Uploaded trained model to https://huggingface.co/{SAVE_REPO_TYPE}s/{SAVE_REPO_ID}/tree/main/{SAVE_PATH_IN_REPO}")


if __name__ == "__main__":
    main()
