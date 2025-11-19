"""
Example T5 Training Script for EEG Report Multi-Task Dataset

This script demonstrates how to train T5 on the generated EEG report dataset.
You can adapt this for your specific needs.

Usage:
    # Single GPU Training
    # Phase 1: DAPT (Domain-Adaptive Pretraining)
    python train_t5_example.py --phase dapt --epochs 5
    
    # Phase 2: Multi-Task SFT (Supervised Fine-Tuning)
    python train_t5_example.py --phase sft --epochs 3 --model ./results/dapt_final
    
    # With Weights & Biases monitoring
    python train_t5_example.py --phase dapt --epochs 5 --use_wandb --wandb_project my-eeg-project --wandb_run_name dapt_v1
    
    # With both W&B and TensorBoard
    python train_t5_example.py --phase dapt --epochs 5 --use_wandb --use_tensorboard
    
    # Multi-GPU Training (DDP)
    # Using torchrun (recommended for PyTorch >= 1.10)
    torchrun --nproc_per_node=4 train_t5_example.py --phase dapt --epochs 5 --distributed --use_wandb
    
    # Using torch.distributed.launch (older PyTorch versions)
    python -m torch.distributed.launch --nproc_per_node=4 train_t5_example.py --phase dapt --epochs 5 --distributed --use_wandb
    
    # Multi-node training
    # Node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=12355 train_t5_example.py --phase dapt --epochs 5 --distributed --world_size=8 --use_wandb
    
    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=12355 train_t5_example.py --phase dapt --epochs 5 --distributed --world_size=8 --use_wandb

Weights & Biases Setup:
    1. Install wandb: pip install wandb
    2. Login: wandb login
    3. Add --use_wandb flag to enable monitoring
    4. Optional: customize with --wandb_project, --wandb_run_name, --wandb_notes
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def setup_wandb(args, phase):
    """Initialize Weights & Biases for experiment tracking"""
    if not WANDB_AVAILABLE or not args.use_wandb:
        return False
    
    # Only initialize on main process in distributed training
    if args.local_rank not in [-1, 0]:
        return False
    
    # Create run name
    run_name = args.wandb_run_name or f"{phase}_{args.model.split('/')[-1]}_e{args.epochs}_bs{args.batch_size}"
    
    # Configuration to log
    config = {
        'phase': phase,
        'model': args.model,
        'tokenizer': args.tokenizer,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 2e-4 if phase == 'dapt' else 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 1000 if phase == 'dapt' else 500,
        'gradient_accumulation_steps': 2,
        'max_input_length': 512,
        'max_output_length': 256,
        'distributed': args.local_rank != -1,
        'world_size': args.world_size if hasattr(args, 'world_size') else 1,
    }
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=config,
        tags=[phase, args.model.split('/')[-1]],
        notes=args.wandb_notes,
        resume='allow' if args.wandb_resume else False,
    )
    
    print(f"✓ Weights & Biases initialized")
    print(f"  Project: {args.wandb_project}")
    print(f"  Run: {run_name}")
    print(f"  URL: {wandb.run.get_url()}")
    
    return True


def setup_distributed(args):
    if args.distributed or args.local_rank != -1:
        # auto fill if not provided
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank if args.local_rank != -1 else 0))
        args.rank = int(os.environ.get('RANK', args.local_rank))
        args.world_size = int(os.environ.get('WORLD_SIZE', args.world_size if hasattr(args, 'world_size') else 1))
        args.master_addr = os.environ.get('MASTER_ADDR', getattr(args, 'master_addr', '127.0.0.1'))
        args.master_port = os.environ.get('MASTER_PORT', getattr(args, 'master_port', '29500'))

        # set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device('cuda', args.local_rank)
        else:
            args.device = torch.device('cpu')

        if not dist.is_initialized():
            dist.init_process_group(
                backend=args.backend if torch.cuda.is_available() else 'gloo',
                init_method=f"tcp://{args.master_addr}:{args.master_port}",
                world_size=args.world_size,
                rank=args.rank,
            )
        print(f"[Rank {args.rank}] Initialized distributed training "
              f"on {args.master_addr}:{args.master_port} "
              f"({args.rank}/{args.world_size})")
        return True
    else:
        args.local_rank = -1
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return False


class EEGReportDataset(Dataset):
    """Dataset for EEG report multi-task learning"""
    
    def __init__(
        self,
        jsonl_file: str,
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_output_length: int = 256,
        task_filter: str = None,
        add_task_prefix: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.add_task_prefix = add_task_prefix
        
        # Load samples
        print(f"Loading samples from {jsonl_file}...")
        self.samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                # Filter by task if specified
                if task_filter is None or sample['task'] == task_filter:
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples")
        if task_filter:
            print(f"  (filtered for task: {task_filter})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Prepare input with optional task prefix
        if self.add_task_prefix:
            input_text = f"{sample['task']}: {sample['input']}"
        else:
            input_text = sample['input']
        
        output_text = sample['output']
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        labels = output_encoding['input_ids']
        # Replace padding token id with -100 so it's ignored by loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        }


def train_dapt(args):
    """Phase 1: Domain-Adaptive Pretraining"""
    print("\n" + "=" * 80)
    print("Phase 1: Domain-Adaptive Pretraining (DAPT)")
    print("=" * 80)
    print(f"Task: pretrain_span_corrupt only")
    print(f"Objective: Adapt T5 to EEG report language distribution")
    print()
    
    # Initialize wandb
    setup_wandb(args, 'dapt')
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    if "[REDACTED]" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["[REDACTED]"])
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    
    # Resize model embeddings if using custom tokenizer
    if args.tokenizer and tokenizer.vocab_size != model.config.vocab_size:
        # print(f"Resizing model embeddings: {model.config.vocab_size} -> {tokenizer.vocab_size}")
        # model.resize_token_embeddings(len(tokenizer))
        extra_ids = 100
        new_size = tokenizer.vocab_size + extra_ids
        print(f"Resizing model embeddings to include extra_ids: {tokenizer.vocab_size} -> {new_size}")
        model.resize_token_embeddings(new_size)

        # print("Tokenizer vocab size:", tokenizer.vocab_size)
        # print("Model vocab size:", model.config.vocab_size)
        # print("Pad token:", tokenizer.pad_token, "->", tokenizer.pad_token_id)
        # print("EOS token:", tokenizer.eos_token, "->", tokenizer.eos_token_id)
        # print("UNK token:", tokenizer.unk_token, "->", tokenizer.unk_token_id)
        # print("Decoder start token:", model.config.decoder_start_token_id)

    # Load datasets (only pretrain_span_corrupt task)
    train_dataset = EEGReportDataset(
        'processed_llm_mp/train.jsonl',
        tokenizer,
        max_input_length=512,
        max_output_length=256,
        task_filter='pretrain_span_corrupt',
        add_task_prefix=False,  # No task prefix for pretraining
    )
    # first_batch = next(iter(train_dataset))
    # print("Max input id:", max(first_batch["input_ids"]))
    # print("Max label id:", max(first_batch["labels"]))
    # print("Vocab size:", tokenizer.vocab_size)

    valid_dataset = EEGReportDataset(
        'processed_llm_mp/valid.jsonl',
        tokenizer,
        max_input_length=512,
        max_output_length=256,
        task_filter='pretrain_span_corrupt',
        add_task_prefix=False,
    )
    
    # Training arguments
    report_to = []
    if args.use_wandb and WANDB_AVAILABLE:
        report_to.append('wandb')
    if args.use_tensorboard:
        report_to.append('tensorboard')
    if not report_to:
        report_to = ['none']
    
    training_args_dict = {
        'output_dir': args.output_dir or './results/dapt',
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'logging_steps': 100,
        'save_steps': 5000,
        'eval_steps': 5000,
        'eval_strategy': 'steps',
        'save_total_limit': 3,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'bf16': torch.cuda.is_available(),
        'gradient_accumulation_steps': 2,
        'report_to': report_to,
    }
    
    # Add DDP settings only if using distributed training
    if args.local_rank != -1:
        training_args_dict['local_rank'] = args.local_rank
        training_args_dict['ddp_backend'] = 'nccl' if torch.cuda.is_available() else 'gloo'
        training_args_dict['ddp_find_unused_parameters'] = False
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_output = Path(training_args.output_dir) / 'final'
    trainer.save_model(str(final_output))
    print(f"\nSaved final model to: {final_output}")


def train_sft(args):
    """Phase 2: Multi-Task Supervised Fine-Tuning"""
    print("\n" + "=" * 80)
    print("Phase 2: Multi-Task Supervised Fine-Tuning (SFT)")
    print("=" * 80)
    print(f"Tasks: All (pretrain, summarize, normalize, polish)")
    print(f"Objective: Learn specific clinical tasks")
    print()
    
    # Initialize wandb
    setup_wandb(args, 'sft')
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    
    # Resize model embeddings if using custom tokenizer
    if args.tokenizer and tokenizer.vocab_size != model.config.vocab_size:
        extra_ids = 100
        new_size = tokenizer.vocab_size + extra_ids
        print(f"Resizing model embeddings to include extra_ids: {tokenizer.vocab_size} -> {new_size}")
        model.resize_token_embeddings(new_size)

    # Load datasets (all tasks)
    train_dataset = EEGReportDataset(
        'processed_llm_mp/train.jsonl',
        tokenizer,
        max_input_length=512,
        max_output_length=256,
        task_filter=None,  # All tasks
        add_task_prefix=True,  # Add task prefix for multi-task learning
    )
    
    valid_dataset = EEGReportDataset(
        'processed_llm_mp/valid.jsonl',
        tokenizer,
        max_input_length=512,
        max_output_length=256,
        task_filter=None,
        add_task_prefix=True,
    )
    
    # Training arguments
    report_to = []
    if args.use_wandb and WANDB_AVAILABLE:
        report_to.append('wandb')
    if args.use_tensorboard:
        report_to.append('tensorboard')
    if not report_to:
        report_to = ['none']
    
    training_args_dict = {
        'output_dir': args.output_dir or './results/sft',
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'logging_steps': 100,
        'save_steps': 5000,
        'eval_steps': 5000,
        'eval_strategy': 'steps',
        'save_total_limit': 3,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'bf16': torch.cuda.is_available(),
        'gradient_accumulation_steps': 2,
        'report_to': report_to,
    }
    
    # Add DDP settings only if using distributed training
    if args.local_rank != -1:
        training_args_dict['local_rank'] = args.local_rank
        training_args_dict['ddp_backend'] = 'nccl' if torch.cuda.is_available() else 'gloo'
        training_args_dict['ddp_find_unused_parameters'] = False
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_output = Path(training_args.output_dir) / 'final'
    trainer.save_model(str(final_output))
    print(f"\nSaved final model to: {final_output}")


def main():
    parser = argparse.ArgumentParser(
        description='Train T5 on EEG Report Multi-Task Dataset'
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['dapt', 'sft'],
        required=True,
        help='Training phase: dapt (domain-adaptive pretraining) or sft (supervised fine-tuning)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='t5-base',
        help='Model name or path (default: t5-base)'
    )
    
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='Custom tokenizer path (default: use model tokenizer)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per device (default: 16)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: ./results/{phase})'
    )
    
    # DDP (Distributed Data Parallel) arguments
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training (automatically set by torch.distributed.launch)'
    )
    
    parser.add_argument(
        '--world_size',
        type=int,
        default=1,
        help='Number of processes for distributed training (default: 1)'
    )
    
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed training with DDP'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        choices=['nccl', 'gloo', 'mpi'],
        help='Distributed backend (default: nccl for GPU, gloo for CPU)'
    )
    
    parser.add_argument(
        '--master_addr',
        type=str,
        default='127.0.0.1',
        help='Master node address for distributed training (default: localhost)'
    )
    
    parser.add_argument(
        '--master_port',
        type=str,
        default='12355',
        help='Master node port for distributed training (default: 12355)'
    )
    
    # Weights & Biases arguments
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='eeg-t5-training',
        help='Weights & Biases project name (default: eeg-t5-training)'
    )
    
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Weights & Biases run name (default: auto-generated)'
    )
    
    parser.add_argument(
        '--wandb_notes',
        type=str,
        default=None,
        help='Notes to add to the Weights & Biases run'
    )
    
    parser.add_argument(
        '--wandb_resume',
        action='store_true',
        help='Resume from a previous Weights & Biases run if available'
    )
    
    parser.add_argument(
        '--use_tensorboard',
        action='store_true',
        help='Enable TensorBoard logging (can be used with wandb)'
    )
    
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed = setup_distributed(args)
    
    # Only main process prints info
    if args.local_rank in [-1, 0]:
        print(f"Training mode: {'Distributed (DDP)' if is_distributed else 'Single Process'}")
        if is_distributed:
            print(f"World size: {args.world_size}, Local rank: {args.local_rank}")
        print(f"Device: {args.device}")
        print()
    
    # Check if dataset exists
    if not Path('processed_llm_mp/train.jsonl').exists():
        print("Error: Dataset not found!")
        print("Please run: python preprocess.py")
        return
    
    # Run appropriate training phase
    if args.phase == 'dapt':
        train_dapt(args)
    else:  # sft
        train_sft(args)
    
    # Cleanup distributed training
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE and args.local_rank in [-1, 0]:
        wandb.finish()
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

