# NeuroLex

NeuroLex is a lightweight pipeline for turning raw, structured EEG reports into multi-task datasets that can train T5-style language models. It bundles:

- **LLM-assisted preprocessing** (`preprocess_llm_multiprocess.py`) that parses each report, anonymizes text, and calls OpenAI models in parallel to create span-corruption, polishing, summarization, QA, and information-extraction samples.
- **Domain tokenizer training** (`train_eeg_tokenizer.py`) for building a SentencePiece vocabulary that better handles EEG terminology.
- **T5 training scaffolding** (`train_t5_example.py`) that runs Domain-Adaptive Pretraining (DAPT) and Supervised Fine-Tuning (SFT) on the generated JSONL datasets with optional Weights & Biases and TensorBoard logging.

The repository is MIT licensed and designed to be copy-pasted into larger clinical NLP workflows.

## Repository layout
- `preprocess_llm_multiprocess.py` – multiprocessing data builder that reads `.txt` EEG reports and emits per-task and combined JSONL splits.
- `train_eeg_tokenizer.py` – collects text from reports, trains a SentencePiece model, converts it to Hugging Face format, and prints tokenization stats.
- `train_t5_example.py` – end-to-end Hugging Face `Trainer` script with single-node and DDP examples for both DAPT and SFT phases.
- `requirements.txt` – consolidated Python dependencies for preprocessing, tokenizer training, and model training.

## Prerequisites
1. **Python**: 3.9+ (tested with 3.10).
2. **Hardware**: CPU for preprocessing/tokenizer steps; CUDA GPU(s) advised for T5 training.
3. **Virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **OpenAI API access** for the preprocessing step:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Preparing raw EEG reports
The preprocessor expects `.txt` files under a directory such as `./eeg_reports` (recursively scanned). Each file should be structured with 80-character separators and upper-cased section headers, for example:

```
================================================================================
IMPRESSION:
================================================================================

Background demonstrates posterior 10 Hz alpha rhythm that attenuates with eye opening...
```

Any category that contains clinically meaningful prose (e.g., `BACKGROUND`, `CONTINUOUS_EEG`, `IMPRESSION`) will be sampled. Empty sections or `NA` entries are discarded automatically.

## LLM-assisted preprocessing
Run the multiprocessing builder to convert raw reports into multi-task JSONL datasets:

```bash
python preprocess_llm_multiprocess.py \
  --report_dir ./eeg_reports \
  --output_dir ./processed_llm_mp \
  --model gpt-4o-mini \
  --num_workers 8 \
  --max_reports 2000 \
  --tasks span_corrupt,polish,summarize,qa,ie
```

Key flags:

| Flag | Description |
| --- | --- |
| `--report_dir` | Root directory containing `.txt` EEG reports. |
| `--output_dir` | Destination for checkpoints plus `{task}/train|valid|test.jsonl` and combined `train/valid/test.jsonl`. |
| `--api_key` | Override for `OPENAI_API_KEY`. |
| `--model` | Any chat-completions model supported by the `openai` Python SDK (default `gpt-4.1-nano`). |
| `--num_workers` | Number of worker processes (defaults to CPU core count). |
| `--tasks` | Comma-separated subset of `span_corrupt`, `polish`, `summarize`, `qa`, `ie`; use `all` for every task. |
| `--only_ie` | Shortcut to run only the information-extraction task. |
| `--max_reports` | Cap how many reports are processed (helpful for dry runs). |
| `--save_every` | Batch size for checkpointing partially processed samples. |

Outputs:
- `processed_llm_mp/checkpoints/checkpoint_*.jsonl` – rolling snapshots so you can resume if interrupted.
- `processed_llm_mp/<task>/train.jsonl`, `valid.jsonl`, `test.jsonl` – task-specific splits.
- `processed_llm_mp/train.jsonl`, `valid.jsonl`, `test.jsonl` – shuffled mixture spanning all tasks with unified schema `{ "task": str, "input": str, "output": str }`.

💡 **Cost control**: Start with `--max_reports 50 --num_workers 2` to validate the pipeline before scaling up API usage.

## Training a domain-specific tokenizer
Use the tokenizer script to build a SentencePiece model from the same reports, then convert it to Hugging Face format:

```bash
python train_eeg_tokenizer.py \
  --report_dir ./eeg_reports \
  --vocab_size 32000 \
  --output_dir ./eeg_tokenizer \
  --base_model t5-base
```

This command:
1. Aggregates text from every report into `./eeg_tokenizer/training_text.txt`.
2. Fits a SentencePiece model (`eeg_sp.model` / `eeg_sp.vocab`) with 100 `<extra_id_*>` tokens, mirroring T5.
3. Saves a Hugging Face-compatible tokenizer under `./eeg_tokenizer/transformers`.
4. Prints tokenization comparisons between the original `t5-base` tokenizer and your custom tokenizer for several EEG sentences.

You can then pass `--tokenizer ./eeg_tokenizer/transformers` to the T5 training script.

## Training T5 (DAPT + SFT)
The example script assumes the combined dataset lives at `./processed_llm_mp/train|valid|test.jsonl`.

### Phase 1 – Domain-Adaptive Pretraining (DAPT)
```bash
python train_t5_example.py \
  --phase dapt \
  --model t5-base \
  --tokenizer ./eeg_tokenizer/transformers \
  --epochs 5 \
  --batch_size 32 \
  --output_dir ./results/dapt \
  --use_wandb --wandb_project neurolex --wandb_run_name dapt_v1
```

### Phase 2 – Supervised Fine-Tuning (SFT)
Use the best checkpoint from DAPT (e.g., `./results/dapt/final`):
```bash
python train_t5_example.py \
  --phase sft \
  --model ./results/dapt/final \
  --tokenizer ./eeg_tokenizer/transformers \
  --epochs 3 \
  --batch_size 16 \
  --output_dir ./results/sft
```

### Distributed and monitoring options
- **Multi-GPU single node**: `torchrun --nproc_per_node=4 train_t5_example.py --phase dapt --distributed`.
- **Multi-node**: configure `--nnodes`, `--node_rank`, `--master_addr`, and `--master_port` via `torchrun`.
- **Weights & Biases**: Add `--use_wandb` plus optional `--wandb_project`, `--wandb_run_name`, `--wandb_notes`, `--wandb_resume`.
- **TensorBoard**: Append `--use_tensorboard` and point TensorBoard at the `output_dir`.

The script automatically initializes PyTorch Distributed Data Parallel (DDP), logs dataset sizes, and saves the final model to `<output_dir>/final`.

## Troubleshooting tips
- **NLTK punkt download**: The preprocessor downloads `punkt` on first run; ensure your environment has internet access or pre-download with `python -m nltk.downloader punkt`.
- **OpenAI rate limits**: Reduce `--num_workers`, add sleeps, or shard reports per API key when encountering `RateLimitError`.
- **Partial datasets**: Use checkpoint files in `processed_llm_mp/checkpoints` to resume interrupted runs or to inspect intermediate JSONL samples.
- **GPU memory**: Lower `--batch_size` or enable gradient accumulation (tweak `TrainingArguments` inside `train_t5_example.py`) when training large models.

## License
This project is released under the [MIT License](LICENSE).
