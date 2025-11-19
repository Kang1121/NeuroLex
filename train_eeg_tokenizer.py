"""
Train a Domain-Specific Tokenizer for EEG Reports

This script trains a SentencePiece tokenizer on EEG reports to better handle
domain-specific terminology. The trained tokenizer can replace T5's default
tokenizer for improved tokenization of clinical EEG vocabulary.

Usage:
    python train_eeg_tokenizer.py --vocab_size 32000 --output_dir ./eeg_tokenizer
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import sentencepiece as spm
from transformers import T5Tokenizer


def collect_text_from_reports(report_dir: str, output_file: str, max_reports: int = None):
    """
    Collect all text from EEG reports into a single training file.
    
    Args:
        report_dir: Directory containing EEG reports
        output_file: Output text file for tokenizer training
        max_reports: Maximum number of reports to process (None = all)
    """
    print(f"Collecting text from: {report_dir}")
    
    report_dir = Path(report_dir)
    all_reports = list(report_dir.rglob("*.txt"))
    
    if max_reports:
        all_reports = all_reports[:max_reports]
    
    print(f"Found {len(all_reports)} reports")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for report_file in tqdm(all_reports, desc="Processing reports"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Remove separator lines
                    content = content.replace('=' * 80, '')
                    
                    # Remove category headers (but keep the actual text)
                    categories = [
                        'BACKGROUND:', 'SLEEP:', 'HYPERVENTILATION:',
                        'PHOTIC_STIMULATION:', 'ABNORMALITY:', 'CARDIAC_MONITOR:',
                        'OXYGEN_MONITOR:', 'CONTINUOUS_EEG:', 'PUSHBUTTON_ACTIVATIONS:',
                        'SEIZURE_DETECTION:', 'QUANTITATIVE_EEG:', 'IMPRESSION:',
                        'ROUTINE_SAMPLING:'
                    ]
                    for cat in categories:
                        content = content.replace(cat, '')
                    
                    # Remove "NA" entries
                    lines = [line.strip() for line in content.split('\n') 
                            if line.strip() and line.strip() != 'NA']
                    
                    # Write non-empty lines
                    for line in lines:
                        if line:
                            out.write(line + '\n')
                            
            except Exception as e:
                print(f"Error processing {report_file}: {e}")
                continue
    
    print(f"Text collected and saved to: {output_file}")


def train_sentencepiece_model(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = 'unigram'
):
    """
    Train a SentencePiece model.
    
    Args:
        input_file: Input text file
        model_prefix: Prefix for output model files
        vocab_size: Size of vocabulary
        character_coverage: Character coverage (higher = more characters)
        model_type: Model type (unigram, bpe, char, word)
    """
    print(f"\nTraining SentencePiece model...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")
    print(f"  Character coverage: {character_coverage}")
    
    # SentencePiece training parameters
    # Following T5's tokenizer configuration
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,  # T5 doesn't use BOS
        pad_piece='<pad>',
        eos_piece='</s>',
        unk_piece='<unk>',
        # Extra IDs for T5 (used for span corruption)
        control_symbols=['<extra_id_0>', '<extra_id_1>', '<extra_id_2>', '<extra_id_3>', 
                        '<extra_id_4>', '<extra_id_5>', '<extra_id_6>', '<extra_id_7>',
                        '<extra_id_8>', '<extra_id_9>'] + 
                       [f'<extra_id_{i}>' for i in range(10, 100)],  # T5 uses 100 extra_ids
        user_defined_symbols=[],
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,  # Handle any byte sequence
    )
    
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocab saved to: {model_prefix}.vocab")


def convert_to_transformers_format(
    sentencepiece_model: str,
    output_dir: str,
    base_t5_model: str = 't5-base'
):
    """
    Convert SentencePiece model to Transformers-compatible format.
    
    Args:
        sentencepiece_model: Path to .model file
        output_dir: Output directory for Transformers tokenizer
        base_t5_model: Base T5 model to copy config from
    """
    print(f"\nConverting to Transformers format...")
    
    # Load base T5 tokenizer to get the configuration
    base_tokenizer = T5Tokenizer.from_pretrained(base_t5_model)
    
    # Create new tokenizer with our SentencePiece model
    tokenizer = T5Tokenizer(
        vocab_file=sentencepiece_model,
        eos_token=base_tokenizer.eos_token,
        unk_token=base_tokenizer.unk_token,
        pad_token=base_tokenizer.pad_token,
        extra_ids=100,  # T5 uses 100 extra IDs
        additional_special_tokens=base_tokenizer.additional_special_tokens,
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Tokenizer saved to: {output_dir}")
    
    # Save metadata
    metadata = {
        'base_model': base_t5_model,
        'vocab_size': tokenizer.vocab_size,
        'model_max_length': tokenizer.model_max_length,
        'special_tokens': {
            'eos_token': tokenizer.eos_token,
            'unk_token': tokenizer.unk_token,
            'pad_token': tokenizer.pad_token,
        },
        'num_extra_ids': 100,
    }
    
    with open(os.path.join(output_dir, 'tokenizer_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {output_dir}/tokenizer_metadata.json")


def analyze_tokenization(
    original_tokenizer_name: str,
    custom_tokenizer_dir: str,
    sample_texts: List[str]
):
    """
    Compare tokenization between original and custom tokenizers.
    
    Args:
        original_tokenizer_name: Name of original tokenizer (e.g., 't5-base')
        custom_tokenizer_dir: Directory of custom tokenizer
        sample_texts: List of sample texts to analyze
    """
    print("\n" + "=" * 80)
    print("Tokenization Comparison")
    print("=" * 80)
    
    orig_tokenizer = T5Tokenizer.from_pretrained(original_tokenizer_name)
    custom_tokenizer = T5Tokenizer.from_pretrained(custom_tokenizer_dir)
    
    print(f"\nOriginal Tokenizer ({original_tokenizer_name}):")
    print(f"  Vocab size: {orig_tokenizer.vocab_size}")
    
    print(f"\nCustom Tokenizer (EEG-specific):")
    print(f"  Vocab size: {custom_tokenizer.vocab_size}")
    
    print("\n" + "-" * 80)
    print("Sample Tokenizations:")
    print("-" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n[Sample {i}]: {text}")
        
        # Original tokenization
        orig_tokens = orig_tokenizer.tokenize(text)
        orig_ids = orig_tokenizer.encode(text)
        print(f"\n  Original ({len(orig_tokens)} tokens):")
        print(f"    Tokens: {orig_tokens}")
        print(f"    IDs: {orig_ids[:20]}{'...' if len(orig_ids) > 20 else ''}")
        
        # Custom tokenization
        custom_tokens = custom_tokenizer.tokenize(text)
        custom_ids = custom_tokenizer.encode(text)
        print(f"\n  Custom ({len(custom_tokens)} tokens):")
        print(f"    Tokens: {custom_tokens}")
        print(f"    IDs: {custom_ids[:20]}{'...' if len(custom_ids) > 20 else ''}")
        
        # Compare
        token_diff = len(orig_tokens) - len(custom_tokens)
        if token_diff > 0:
            print(f"\n  ✓ Custom tokenizer: {token_diff} fewer tokens ({token_diff/len(orig_tokens)*100:.1f}% reduction)")
        elif token_diff < 0:
            print(f"\n  ✗ Custom tokenizer: {abs(token_diff)} more tokens")
        else:
            print(f"\n  = Same number of tokens")


def main():
    parser = argparse.ArgumentParser(
        description='Train a domain-specific tokenizer for EEG reports'
    )
    
    parser.add_argument(
        '--report_dir',
        type=str,
        default='eeg_reports_eeg_only_13_categories_structured',
        help='Directory containing EEG reports (default: eeg_reports_eeg_only_13_categories_structured)'
    )
    
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=12000,
        help='Vocabulary size (default: 32000, same as T5-base)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./eeg_tokenizer',
        help='Output directory for trained tokenizer (default: ./eeg_tokenizer)'
    )
    
    parser.add_argument(
        '--base_model',
        type=str,
        default='t5-base',
        help='Base T5 model to get config from (default: t5-base)'
    )
    
    parser.add_argument(
        '--max_reports',
        type=int,
        default=None,
        help='Maximum number of reports to use for training (default: all)'
    )
    
    parser.add_argument(
        '--character_coverage',
        type=float,
        default=0.9995,
        help='Character coverage for SentencePiece (default: 0.9995)'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='unigram',
        choices=['unigram', 'bpe', 'char', 'word'],
        help='SentencePiece model type (default: unigram, same as T5)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Collect text from reports
    text_file = os.path.join(args.output_dir, 'training_text.txt')
    if not os.path.exists(text_file):
        collect_text_from_reports(
            args.report_dir,
            text_file,
            max_reports=args.max_reports
        )
    else:
        print(f"Using existing text file: {text_file}")
    
    # Step 2: Train SentencePiece model
    model_prefix = os.path.join(args.output_dir, 'eeg_sp')
    if not os.path.exists(f"{model_prefix}.model"):
        train_sentencepiece_model(
            text_file,
            model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type=args.model_type
        )
    else:
        print(f"Using existing SentencePiece model: {model_prefix}.model")
    
    # Step 3: Convert to Transformers format
    tokenizer_dir = os.path.join(args.output_dir, 'transformers')
    convert_to_transformers_format(
        f"{model_prefix}.model",
        tokenizer_dir,
        base_t5_model=args.base_model
    )
    
    # Step 4: Analyze tokenization on sample texts
    sample_texts = [
        "Intermittent moderate voltage delta activity was seen from the left mid temporal region.",
        "The background was a posteriorly predominant 11 Hz rhythm which attenuated to eye opening.",
        "Abnormal awake EEG due to intermittent left mid temporal and posterior quadrant delta slowing.",
        "Epileptiform discharges with focal sharp waves in the right frontal region.",
        "Photic stimulation produced symmetric driving response without activation.",
        "Continuous EEG monitoring showed no electrographic seizures during the recording period.",
    ]
    
    analyze_tokenization(
        args.base_model,
        tokenizer_dir,
        sample_texts
    )
    
    print("\n" + "=" * 80)
    print("Tokenizer Training Complete!")
    print("=" * 80)
    print(f"\nYour custom EEG tokenizer is ready at: {tokenizer_dir}")
    print(f"\nTo use it in training, run:")
    print(f"  python train_t5_example.py --phase dapt --epochs 5 --tokenizer {tokenizer_dir}")
    print()


if __name__ == '__main__':
    main()

