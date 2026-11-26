"""Random-init OLMo 2 baseline for prepacked, fixed-length token blocks."""

import argparse
from pathlib import Path
import warnings

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed


class StepLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs and "loss" in logs:
            loss = logs["loss"]
            if not torch.isfinite(torch.tensor(loss)):
                raise RuntimeError(f"Nonfinite training loss at step {state.global_step}")
            print(f"step:{state.global_step}/{state.max_steps} train_loss:{loss}", flush=True)


def collate(examples):
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets", required=True, help="Local model config and tokenizer directory")
    cli.add_argument("--train", required=True, help="Local datasets.save_to_disk directory with input_ids")
    cli.add_argument("--out", required=True, type=Path, help="Final Hugging Face checkpoint directory")
    cli.add_argument("--steps", type=int, default=2000)
    cli.add_argument("--sequence-length", type=int, default=2048)
    cli.add_argument("--micro-batch", type=int, default=4)
    cli.add_argument("--accumulation", type=int, default=4)
    cli.add_argument("--lr", type=float, default=3e-4)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    if min(args.steps, args.sequence_length, args.micro_batch, args.accumulation) < 1:
        cli.error("Steps, sequence length and batch settings must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        cli.error("Choose an empty checkpoint directory")

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.assets, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.assets, local_files_only=True, trust_remote_code=False)
    config.use_cache = False
    if args.sequence_length > config.max_position_embeddings:
        cli.error("Sequence length exceeds the model context")
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    data = load_from_disk(args.train)
    if not isinstance(data, Dataset) or not len(data) or "input_ids" not in data.column_names:
        cli.error("Expected a non-empty Dataset containing input_ids")
    if len(data[0]["input_ids"]) != args.sequence_length:
        cli.error("Packed dataset sequence length does not match --sequence-length")

    training = TrainingArguments(
        output_dir=str(args.out),
        max_steps=args.steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.accumulation,
        dataloader_drop_last=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        weight_decay=0.1,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="no",
        save_safetensors=True,
        disable_tqdm=True,
        report_to=[],
        seed=args.seed,
    )
    global_batch = training.world_size * args.micro_batch * args.accumulation
    usable_rows = len(data) // global_batch * global_batch
    if not usable_rows:
        cli.error("Packed dataset must contain at least one full global batch")
    data = data.select(range(usable_rows))
    if training.world_size != 8 and torch.cuda.is_available() and training.process_index == 0:
        warnings.warn("The course baseline is configured for eight GPUs. This run uses a different GPU count.")
    trainer = Trainer(model=model, args=training, train_dataset=data, data_collator=collate, callbacks=[StepLoss()])
    if trainer.is_world_process_zero():
        print(f"Parameters: {model.num_parameters():,} | world_size={training.world_size}", flush=True)
    trainer.train()
    trainer.model.config.use_cache = True
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
