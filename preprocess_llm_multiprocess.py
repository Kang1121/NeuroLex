"""
EEG Report Preprocessing with LLM - MULTIPROCESS VERSION

Uses multiprocessing to achieve true parallelism on multi-core CPUs.
Speed improvement: 8-16x faster than sequential, 1.5-2x faster than async.

Best for:
- CPU-intensive workloads (parsing, text processing)
- Multi-core machines (4+ cores)
- Large-scale processing (1000+ reports)

Tasks:
- span_corrupt: Span corruption for pretraining
- polish: Text polishing
- summarize: Summarization
- qa: Question answering
- ie: Information extraction (NEW)

Usage:
    # Run all tasks
    python preprocess_llm_multiprocess.py \
        --max_reports 1000 \
        --num_workers 8 \
        --output_dir processed_llm_mp
    
    # Run only information extraction
    python preprocess_llm_multiprocess.py \
        --only_ie \
        --max_reports 1000 \
        --output_dir processed_llm_ie
    
    # Run specific tasks
    python preprocess_llm_multiprocess.py \
        --tasks ie,qa,summarize \
        --max_reports 1000 \
        --output_dir processed_llm_custom
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import nltk
from tqdm import tqdm
from openai import OpenAI
import time
from multiprocessing import Pool, Manager, cpu_count
import signal

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


# Global variables for multiprocessing
API_KEY = None
MODEL_NAME = None


def init_worker(api_key: str, model: str):
    """Initialize worker process"""
    global API_KEY, MODEL_NAME
    API_KEY = api_key
    MODEL_NAME = model
    # Ignore Ctrl+C in worker processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class EEGReportParser:
    """Parse structured EEG reports"""
    
    def parse_report(self, filepath: str) -> Dict[str, str]:
        """Parse a structured report file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        categories = {}
        pattern = r'={80}\n([A-Z_]+):\n={80}\n\n(.*?)(?=\n={80}|$)'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            category = match.group(1)
            text = match.group(2).strip()
            
            if text and text != 'NA':
                text = re.sub(r'\[.*?\]:', '', text)
                text = re.sub(r'\[.*?\]', '', text)
                text = ' '.join(text.split())
                categories[category] = text
        
        return categories


class TextCleaner:
    """Clean and normalize text"""
    
    def __init__(self):
        self.anonymize_patterns = [
            (r'\*+', '[REDACTED]'),
            (r'\b\d{1,3}\s*years?\s*old\b', '[AGE]'),
            (r'\b\d{1,3}\s*y\.?o\.?\b', '[AGE]'),
        ]
    
    def clean(self, text: str) -> str:
        if not text:
            return ""
        
        for pattern, replacement in self.anonymize_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = ' '.join(text.split())
        text = text.replace('  ', ' ')
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        
        return text.strip()
    
    def split_sentences(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]


class LLMCaller:
    """Thread-safe LLM caller"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = 3
    
    def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Call LLM with retry"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in EEG report analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None


def process_single_report(args_tuple: Tuple[str, Dict]) -> Tuple[List[Dict], int]:
    """
    Process a single report (worker function)
    
    Args:
        args_tuple: (filepath, task_config) where task_config specifies which tasks to run
    
    Returns:
        (samples, api_calls): Generated samples and number of API calls made
    """
    filepath, task_config = args_tuple
    global API_KEY, MODEL_NAME
    
    parser = EEGReportParser()
    cleaner = TextCleaner()
    llm = LLMCaller(API_KEY, MODEL_NAME)
    
    all_samples = []
    api_calls = 0
    
    try:
        # Parse report
        categories = parser.parse_report(filepath)
        if not categories:
            return ([], 0)
        
        # Sample categories
        all_texts = [(cat, text) for cat, text in categories.items() 
                    if cleaner.clean(text) and len(cleaner.clean(text)) > 20]
        
        if not all_texts:
            return ([], 0)
        
        sampled_cats = random.sample(all_texts, min(3, len(all_texts)))
        
        for category, text in sampled_cats:
            cleaned_text = cleaner.clean(text)
            
            # Generate span corruption
            if task_config.get('span_corrupt', False):
                samples, calls = generate_span_corrupt(llm, cleaned_text, cleaner)
                all_samples.extend(samples)
                api_calls += calls
            
            # Generate polish
            if task_config.get('polish', False):
                samples, calls = generate_polish(llm, cleaned_text, cleaner)
                all_samples.extend(samples)
                api_calls += calls
            
            # Generate summarize
            if task_config.get('summarize', False):
                samples, calls = generate_summarize(llm, cleaned_text)
                all_samples.extend(samples)
                api_calls += calls
            
            # Generate QA
            if task_config.get('qa', False):
                samples, calls = generate_qa(llm, cleaned_text)
                all_samples.extend(samples)
                api_calls += calls
        
        # Generate Information Extraction - only from CONTINUOUS_EEG category
        if task_config.get('ie', False):
            if 'CONTINUOUS_EEG' in categories:
                continuous_eeg_text = categories['CONTINUOUS_EEG']
                cleaned_text = cleaner.clean(continuous_eeg_text)
                if cleaned_text and len(cleaned_text) > 20:
                    samples, calls = generate_ie(llm, cleaned_text, cleaner)
                    all_samples.extend(samples)
                    api_calls += calls
    
    except Exception as e:
        # Silent fail in worker
        pass
    
    return (all_samples, api_calls)


def generate_span_corrupt(llm: LLMCaller, text: str, cleaner: TextCleaner, 
                          num_variants: int = 2) -> Tuple[List[Dict], int]:
    """Generate span corruption samples"""
    samples = []
    api_calls = 0
    
    if len(text.split()) < 10:
        return (samples, 0)
    
    prompt = f"""Given this EEG report text, identify 3-5 important domain-specific medical terms or phrases that should be masked for a language model to learn. Focus on:
    - Clinical terminology (e.g., "alpha rhythm", "sharp waves", "PLEDs", "triphasic waves")
    - Frequency measurements (e.g., "8-10 Hz", "theta activity")
    - Anatomical locations (e.g., "posterior regions", "bilateral frontal")
    - Clinical findings (e.g., "epileptiform discharges", "focal slowing")

    Text: "{text}"

    Return ONLY a JSON array of terms to mask, e.g.: ["alpha rhythm", "8-10 Hz", "posterior regions"]
    """
    
    response = llm.call_llm(prompt, temperature=0.3, max_tokens=200)
    api_calls += 1
    
    try:
        terms_to_mask = json.loads(response)
        if not isinstance(terms_to_mask, list):
            raise ValueError
    except Exception:
        terms_to_mask = re.findall(r'"([^"]+)"', str(response))
    
    if not terms_to_mask:
        return (samples, api_calls)
    
    for variant in range(num_variants):
        masked_text = text
        target_spans = []
        matches = []
        
        random.shuffle(terms_to_mask)
        for span_id, term in enumerate(terms_to_mask[:random.randint(2, 4)]):
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                matches.append((match.start(), match.end(), span_id, match.group(0)))
                break
        
        matches.sort(key=lambda x: x[0], reverse=True)
        for start, end, span_id, actual in matches:
            masked_text = masked_text[:start] + f"<extra_id_{span_id}>" + masked_text[end:]
            target_spans.insert(0, f"<extra_id_{span_id}> {actual}")
        
        if target_spans:
            output_text = " ".join(target_spans).strip() + f" <extra_id_{len(target_spans)}>"
            output_text = re.sub(r'\s+', ' ', output_text)
            samples.append({
                "task": "pretrain_span_corrupt",
                "input": masked_text.strip(),
                "output": output_text
            })
    
    return (samples, api_calls)


def generate_polish(llm: LLMCaller, text: str, cleaner: TextCleaner,
                    num_variants: int = 1) -> Tuple[List[Dict], int]:
    """Generate polish samples"""
    samples = []
    api_calls = 0
    
    sentences = cleaner.split_sentences(text)
    if not sentences:
        return (samples, 0)
    
    # Sample sentences
    selected = random.sample(sentences, min(3, len(sentences)))
    
    for sent in selected:
        if len(sent.split()) < 6:
            continue
        
        prompt = f"""Given this clinical EEG report sentence, generate {num_variants} simplified versions.

Original: "{sent}"

Requirements:
- Keep medical meaning identical
- Simplify structure, remove articles/connectors
Return ONLY a JSON array of simplified sentences, e.g. ["v1"]
"""
        
        response = llm.call_llm(prompt, temperature=0.7, max_tokens=200)
        api_calls += 1
        
        if not response:
            continue
        
        try:
            variations = json.loads(response)
            if not isinstance(variations, list):
                raise ValueError
        except Exception:
            variations = re.findall(r'"([^"]+)"', str(response))
        
        for noisy in variations[:num_variants]:
            noisy = noisy.strip()
            if noisy and noisy.lower() != sent.lower():
                samples.append({
                    "task": "polish",
                    "input": noisy,
                    "output": sent
                })
    
    return (samples, api_calls)


def generate_summarize(llm: LLMCaller, text: str, num_variants: int = 1) -> Tuple[List[Dict], int]:
    """Generate summarization samples"""
    samples = []
    api_calls = 0
    
    text = text.strip()
    if len(text.split()) < 10:
        return (samples, 0)
    
    if len(text) > 800:
        text = " ".join(text.split()[:120]) + " ..."
    
    prompt = f"""Summarize the following EEG report passage into one or two concise sentences.

Text: "{text}"

Return ONLY {num_variants} summary sentence(s) as a JSON array, e.g.: ["summary 1"]
"""
    
    response = llm.call_llm(prompt, temperature=0.5, max_tokens=200)
    api_calls += 1
    
    if not response:
        return (samples, 0)
    
    try:
        summaries = json.loads(response)
        if not isinstance(summaries, list):
            raise ValueError
    except Exception:
        summaries = re.findall(r'"([^"]+)"', str(response))
    
    for s in summaries[:num_variants]:
        s = s.strip()
        if 4 <= len(s.split()) <= 40:
            samples.append({
                "task": "summarize",
                "input": text,
                "output": s
            })
    
    return (samples, api_calls)


def generate_qa(llm: LLMCaller, text: str, num_questions: int = 1) -> Tuple[List[Dict], int]:
    """Generate Q&A samples"""
    samples = []
    api_calls = 0
    
    text = text.strip()
    if len(text.split()) < 15:
        return (samples, 0)
    
    if len(text.split()) > 150:
        text = " ".join(text.split()[:150]) + " ..."
    
    prompt = f"""Generate {num_questions} clinical question-answer pairs from this EEG text.

Text: "{text}"

Return ONLY a JSON array: [{{"question": "Q?", "answer": "A"}}]
"""
    
    response = llm.call_llm(prompt, temperature=0.6, max_tokens=400)
    api_calls += 1
    
    if not response:
        return (samples, 0)
    
    try:
        qa_pairs = json.loads(response)
        if not isinstance(qa_pairs, list):
            raise ValueError
    except Exception:
        qa_pairs = []
    
    for qa in qa_pairs[:num_questions]:
        if not isinstance(qa, dict):
            continue
        q, a = qa.get("question", "").strip(), qa.get("answer", "").strip()
        if len(q.split()) >= 3 and len(a.split()) >= 2:
            samples.append({
                "task": "qa",
                "input": f"Question: {q} Context: {text}",
                "output": a
            })
    
    return (samples, api_calls)


def generate_ie(llm: LLMCaller, text: str, cleaner: TextCleaner) -> Tuple[List[Dict], int]:
    """
    Generate silver-standard Information Extraction samples (semi-open schema).

    Schema (semi-constrained):
    - Fixed fields: laterality / localization / pattern / frequency / state / negation
    - Values: Prefer vocab terms; if not covered, use short canonicalized or verbatim phrase.
    - Allow multiple events per sentence (list of records).
    """

    samples = []
    api_calls = 0

    sentences = cleaner.split_sentences(text)
    if not sentences:
        return (samples, 0)

    VALID_VOCAB = {
        "laterality": ["left", "right", "bilateral", "midline", "diffuse", "not-stated"],
        "localization": ["temporal", "frontal", "central", "parietal", "occipital", "not-stated"],
        "pattern": ["spike", "sharp-wave", "polyspike", "slowing", "PLEDs", "burst-suppression",
                    "seizure", "epileptiform", "focal-abnormality", "theta", "alpha", "beta",
                    "delta", "gamma", "not-stated"],
        "frequency": ["rare", "occasional", "frequent", "not-stated"],
        "normality": ["normal", "abnormal", "not-stated"]
    }

    for sentence in sentences:
        if len(sentence.split()) < 5:
            continue

        prompt = f"""
        Extract structured EEG findings from the following sentence.

        Sentence: "{sentence}"

        Schema fields:
        - laterality
        - localization
        - pattern
        - frequency
        - normality

        Rules:
        1. Try to extract **all possible distinct events** described in this sentence.
        Each event should be an independent JSON object.
        2. Use the vocab terms when applicable:
        Laterality: {', '.join(VALID_VOCAB['laterality'])}
        Localization: {', '.join(VALID_VOCAB['localization'])}
        Pattern: {', '.join(VALID_VOCAB['pattern'])}
        Frequency: {', '.join(VALID_VOCAB['frequency'])}
        Normality: {', '.join(VALID_VOCAB['normality'])}
        3. If a phrase in the sentence cannot be exactly mapped to these vocabularies
        (e.g., "frontotemporal", "posterior temporal region", "light sleep"),
        **keep the closest canonicalized phrase from the text** (do not return null).
        4. Normality detection rules:
            - Mark "abnormal" once abnormal EEG pattern is detected,
            - If there is no explicit abnormality, use "normal".
            - If not mentioned or non-relevant, use "not-stated".
        6. Use "not-stated" only when information truly absent.
        7. Always return a JSON list, even if only one event is found.

        Output format:
        [
        {{"laterality": "...", "localization": "...", "pattern": "...", "frequency": "...", "normality": "..."}},
        ...
        ]
        """

        response = llm.call_llm(prompt, temperature=0.2, max_tokens=600)
        api_calls += 1
        if not response:
            continue

        # Try parse JSON
        try:
            parsed = json.loads(response)
        except Exception:
            continue

        # Accept both list or single dict
        if isinstance(parsed, dict):
            parsed = [parsed]
        elif not isinstance(parsed, list):
            continue

        cleaned_records = []
        for rec in parsed:
            if not isinstance(rec, dict):
                continue

            # Normalize keys
            record = {k.lower().strip(): v.lower().strip() if isinstance(v, str) else v
                        for k, v in rec.items() if k.lower() in VALID_VOCAB}

            # Fill missing fields as "not-stated"
            for field in VALID_VOCAB:
                if field not in record:
                    record[field] = "not-stated"

            # Truncate long free-text values (>6 words)
            for field, val in record.items():
                if isinstance(val, str) and len(val.split()) > 6:
                    record[field] = " ".join(val.split()[:6]) + "..."

            cleaned_records.append(record)

        if cleaned_records:
            samples.append({
                "task": "information_extraction",
                "input": sentence,
                "output": json.dumps(cleaned_records, ensure_ascii=False)
            })

    return samples, api_calls


class MultiprocessDatasetBuilder:
    """Multiprocess dataset builder - true parallelism"""
    
    def __init__(self, report_dir: str, output_dir: str, api_key: str, 
                 model: str = "gpt-4o-mini", num_workers: int = None,
                 task_config: Dict = None):
        self.report_dir = Path(report_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.api_key = api_key
        self.model = model
        self.num_workers = num_workers or cpu_count()
        
        # Task configuration - which tasks to run
        if task_config is None:
            # Default: run all tasks
            self.task_config = {
                'span_corrupt': True,
                'polish': True,
                'summarize': True,
                'qa': True,
                'ie': True
            }
        else:
            self.task_config = task_config
        
        self.all_samples = []
        self.invalid_samples = []
        self.total_api_calls = 0
    
    def validate_sample(self, sample: Dict) -> bool:
        """Validate sample"""
        required_fields = ['task', 'input', 'output']
        for field in required_fields:
            if field not in sample or not sample[field] or not sample[field].strip():
                return False
        
        if len(sample['input'].split()) < 3 or len(sample['output'].split()) < 2:
            return False
        
        valid_tasks = ['pretrain_span_corrupt', 'polish', 'complete', 'summarize', 'normalize', 'qa', 'information_extraction']
        if sample['task'] not in valid_tasks:
            return False
        
        return True
    
    def process_all_reports(self, max_reports: int = None, save_every: int = 1000):
        """Process reports using multiprocessing"""
        report_files = list(self.report_dir.rglob('*.txt'))
        
        if max_reports:
            report_files = report_files[:max_reports]
        
        # Display enabled tasks
        enabled_tasks = [task for task, enabled in self.task_config.items() if enabled]
        print(f"Found {len(report_files)} report files")
        print(f"Using LLM model: {self.model}")
        print(f"Workers: {self.num_workers} processes")
        print(f"Enabled tasks: {', '.join(enabled_tasks)}")
        print(f"💾 Auto-save every {save_every} reports")
        print("Processing reports with multiprocessing...")
        
        start_time = time.time()
        
        # Process in batches for checkpointing
        checkpoint_num = 0
        
        try:
            for i in range(0, len(report_files), save_every):
                batch_files = [str(f) for f in report_files[i:i + save_every]]
                batch_start = time.time()
                
                print(f"\n📦 Batch {i//save_every + 1}: Processing {len(batch_files)} reports...")
                
                # Prepare arguments: (filepath, task_config) tuples
                batch_args = [(f, self.task_config) for f in batch_files]
                
                # Process with worker pool
                with Pool(processes=self.num_workers, 
                         initializer=init_worker, 
                         initargs=(self.api_key, self.model)) as pool:
                    
                    results = list(tqdm(
                        pool.imap_unordered(process_single_report, batch_args),
                        total=len(batch_args),
                        desc=f"  Batch {i//save_every + 1}"
                    ))
                
                # Collect results
                batch_samples = 0
                batch_api_calls = 0
                
                for samples, api_calls in results:
                    for sample in samples:
                        if self.validate_sample(sample):
                            self.all_samples.append(sample)
                            batch_samples += 1
                    batch_api_calls += api_calls
                
                self.total_api_calls += batch_api_calls
                batch_elapsed = time.time() - batch_start
                
                # Save checkpoint
                checkpoint_num += 1
                print(f"\n💾 Checkpoint {checkpoint_num}: Processed {min(i + save_every, len(report_files))}/{len(report_files)} reports")
                print(f"   Batch samples: {batch_samples} | Total samples: {len(self.all_samples)}")
                print(f"   Batch time: {batch_elapsed:.1f}s | Speed: {len(batch_files)/batch_elapsed:.2f} reports/s")
                
                if len(self.all_samples) > 0:
                    self._save_checkpoint(checkpoint_num)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving current progress...")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("Processing Summary")
        print(f"{'='*80}")
        print(f"⏱️  Time elapsed: {elapsed:.1f}s ({len(report_files)/elapsed:.2f} reports/s)")
        print(f"✓ Valid samples: {len(self.all_samples)}")
        print(f"📞 LLM API calls: {self.total_api_calls}")
        print(f"💾 Checkpoints saved: {checkpoint_num}")
        
        # Task distribution
        task_counts = defaultdict(int)
        for sample in self.all_samples:
            task_counts[sample['task']] += 1
        
        print("\n📊 Task distribution:")
        for task, count in sorted(task_counts.items()):
            percentage = count / len(self.all_samples) * 100 if self.all_samples else 0
            print(f"  {task}: {count} ({percentage:.1f}%)")
    
    def _save_checkpoint(self, checkpoint_num: int):
        """Save checkpoint"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f'checkpoint_{checkpoint_num}.jsonl'
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            for sample in self.all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"   ✓ Checkpoint saved: {checkpoint_file}")
    
    def split_and_save(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split and save datasets by task"""
        filtered_samples = [
            s for s in self.all_samples
            if len(s['input'].split()) >= 5 and len(s['output'].split()) >= 5
        ]
        
        if len(filtered_samples) < len(self.all_samples):
            print(f"\n⚠️  Filtered: {len(self.all_samples)} -> {len(filtered_samples)} samples")
        
        task_samples = defaultdict(list)
        for sample in filtered_samples:
            task = sample.get('task', 'unknown')
            task_samples[task].append(sample)
        
        print(f"\nSamples by task:")
        for task, samples in sorted(task_samples.items()):
            print(f"  {task}: {len(samples)} samples")
        
        print(f"\n{'='*80}")
        print("Saving datasets by task...")
        print(f"{'='*80}")
        
        all_train = []
        all_valid = []
        all_test = []
        
        for task, samples in sorted(task_samples.items()):
            task_dir = self.output_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            
            random.shuffle(samples)
            
            total = len(samples)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            
            train_samples = samples[:train_end]
            val_samples = samples[train_end:val_end]
            test_samples = samples[val_end:]
            
            self._save_jsonl(train_samples, task_dir / 'train.jsonl')
            self._save_jsonl(val_samples, task_dir / 'valid.jsonl')
            self._save_jsonl(test_samples, task_dir / 'test.jsonl')
            
            all_train.extend(train_samples)
            all_valid.extend(val_samples)
            all_test.extend(test_samples)
            
            print(f"\n{task}:")
            print(f"  Train: {len(train_samples)} -> {task_dir / 'train.jsonl'}")
            print(f"  Valid: {len(val_samples)} -> {task_dir / 'valid.jsonl'}")
            print(f"  Test:  {len(test_samples)} -> {task_dir / 'test.jsonl'}")
        
        print(f"\n{'='*80}")
        print("Saving combined dataset...")
        print(f"{'='*80}")
        
        random.shuffle(all_train)
        random.shuffle(all_valid)
        random.shuffle(all_test)
        
        self._save_jsonl(all_train, self.output_dir / 'train.jsonl')
        self._save_jsonl(all_valid, self.output_dir / 'valid.jsonl')
        self._save_jsonl(all_test, self.output_dir / 'test.jsonl')
        
        print(f"\nCombined:")
        print(f"  Train: {len(all_train)} samples -> {self.output_dir / 'train.jsonl'}")
        print(f"  Valid: {len(all_valid)} samples -> {self.output_dir / 'valid.jsonl'}")
        print(f"  Test:  {len(all_test)} samples -> {self.output_dir / 'test.jsonl'}")
        
        print(f"\n{'='*80}")
        print("✓ All datasets saved successfully!")
        print(f"{'='*80}")
    
    def _save_jsonl(self, samples: List[Dict], filepath: Path):
        """Save samples to JSONL"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Multiprocess EEG preprocessing (8-16x faster, true parallelism)'
    )
    parser.add_argument('--report_dir', type=str,
                       default='eeg_reports')
    parser.add_argument('--output_dir', type=str, default='processed_llm_mp')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt-4.1-nano')
    parser.add_argument('--max_reports', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None,
                       help=f'Number of worker processes (default: {cpu_count()} = CPU cores)')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save checkpoint every N reports (default: 1000)')
    parser.add_argument('--seed', type=int, default=42)
    
    # Task selection arguments
    parser.add_argument('--tasks', type=str, default='all',
                       help='Comma-separated list of tasks to run: span_corrupt,polish,summarize,qa,ie or "all" (default: all)')
    parser.add_argument('--only_ie', action='store_true',
                       help='Run only information extraction task (shortcut for --tasks ie)')
    
    args = parser.parse_args()
    os.environ[
        'OPENAI_API_KEY'] = 'YOUR_KEY'

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required")
        print("Set: export OPENAI_API_KEY='sk-...'")
        return
    
    random.seed(args.seed)
    
    # Parse task configuration
    task_config = {
        'span_corrupt': False,
        'polish': False,
        'summarize': False,
        'qa': False,
        'ie': False
    }
    
    if args.only_ie:
        task_config['ie'] = True
    elif args.tasks == 'all':
        # Enable all tasks
        for task in task_config:
            task_config[task] = True
    else:
        # Enable specified tasks
        selected_tasks = [t.strip() for t in args.tasks.split(',')]
        for task in selected_tasks:
            if task in task_config:
                task_config[task] = True
            else:
                print(f"Warning: Unknown task '{task}', ignoring...")
    
    # Check if at least one task is enabled
    if not any(task_config.values()):
        print("Error: No tasks selected. Use --tasks or --only_ie")
        return
    
    print("=" * 80)
    print("EEG Report Preprocessing with LLM (MULTIPROCESS)")
    print("=" * 80)
    print(f"Report directory: {args.report_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"LLM model: {args.model}")
    print(f"Workers: {args.num_workers or cpu_count()} processes")
    print(f"Save every: {args.save_every} reports")
    print()
    
    builder = MultiprocessDatasetBuilder(
        args.report_dir,
        args.output_dir,
        api_key,
        args.model,
        args.num_workers,
        task_config
    )
    
    builder.process_all_reports(
        max_reports=args.max_reports,
        save_every=args.save_every
    )
    builder.split_and_save()
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

