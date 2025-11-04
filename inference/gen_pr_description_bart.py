#!/usr/bin/env python3
"""
BART PR Description Generation Service (Optimized for NVIDIA GPUs)

This script loads a BART model and generates PR descriptions from input data.
Optimized for maximum speed with batch processing and mixed precision.
Supports A100, L4, and other NVIDIA GPUs.

Usage:
    python gen_pr_description_bart.py --input_path <path> --model_path <path> --output_path <path>
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.cuda.amp import autocast

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


def load_model(model_path: str, use_fp16: bool = True):
    """Load the BART model and tokenizer from the specified path."""
    global model, tokenizer, device
    
    print(f"Loading model from: {model_path}")
    
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Optimize CUDA settings for modern GPUs (A100, L4, etc.)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        dtype=torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Enable gradient checkpointing for memory efficiency (optional)
    # model.gradient_checkpointing_enable()
    
    print(f"Model loaded successfully! (FP16: {use_fp16 and torch.cuda.is_available()})")


def generate_batch(texts, max_input_length=1024, max_output_length=1000, use_fp16=True):
    """Generate model output for a batch of texts using CUDA with FP16."""
    # Filter out invalid texts and keep track of indices
    valid_indices = []
    valid_texts = []
    
    for idx, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_indices.append(idx)
            valid_texts.append(text)
    
    if not valid_texts:
        return [None] * len(texts)
    
    try:
        # Tokenize batch with padding
        inputs = tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=True
        ).to(device)
        
        # Generate with mixed precision if enabled
        with torch.no_grad():
            if use_fp16 and torch.cuda.is_available():
                with autocast():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_output_length,
                        num_beams=1,  # Greedy decoding for speed
                        do_sample=False
                    )
            else:
                outputs = model.generate(
                    **inputs,
                    max_length=max_output_length,
                    num_beams=1,
                    do_sample=False
                )
        
        # Decode outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Map back to original indices
        results = [None] * len(texts)
        for valid_idx, decoded_text in zip(valid_indices, decoded_outputs):
            results[valid_idx] = decoded_text
        
        return results
        
    except Exception as e:
        print(f"Error generating batch: {e}")
        return [None] * len(texts)


def process_dataframe(df: pd.DataFrame, column_name: str, output_col: str = "bart_gen_pr_description", 
                      batch_size: int = 16, max_input_length: int = 1024, 
                      max_output_length: int = 1000, use_fp16: bool = True):
    """Process a DataFrame with batch processing for maximum speed."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    print(f"Processing {len(df)} rows with batch size {batch_size}...")
    device_name = "CUDA" if device.type == "cuda" else "CPU"
    
    # Process in batches
    results = []
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(df), batch_size), desc=f"Generating ({device_name}, FP16={use_fp16})", total=num_batches):
        batch_texts = df[column_name].iloc[i:i+batch_size].tolist()
        batch_results = generate_batch(
            batch_texts, 
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            use_fp16=use_fp16
        )
        results.extend(batch_results)
    
    df[output_col] = results
    
    # Print statistics
    generated_count = df[output_col].notna().sum()
    print(f"\n✓ Generated {generated_count}/{len(df)} summaries successfully")
    
    return df


def main():
    """Main function to run the service."""
    parser = argparse.ArgumentParser(
        description="Generate PR descriptions using BART model (Optimized for NVIDIA GPUs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default batch size
  python gen_pr_description_bart.py --input_path data/input.parquet --model_path models/bart-base --output_path data/output.parquet
  
  # For L4 GPU (24GB) - recommended batch sizes: 16-32
  python gen_pr_description_bart.py -i input.parquet -m ./model -o output.parquet --batch_size 16
  
  # For A100 GPU (40/80GB) - can use larger batches: 32-128
  python gen_pr_description_bart.py -i input.parquet -m ./model -o output.parquet --batch_size 64
  
  # Custom column and output
  python gen_pr_description_bart.py -i input.csv -m ./model -o output.csv --column body --output_column summary
        """
    )
    
    parser.add_argument(
        "--input_path", "-i",
        type=str,
        required=True,
        help="Path to input data file (supports .parquet, .csv, .json)"
    )
    
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        required=True,
        help="Path to the BART model directory"
    )
    
    parser.add_argument(
        "--output_path", "-o",
        type=str,
        required=True,
        help="Path to save output data file (supports .parquet, .csv, .json)"
    )
    
    parser.add_argument(
        "--column",
        type=str,
        default="body",
        help="Name of the column containing text to summarize (default: body)"
    )
    
    parser.add_argument(
        "--output_column",
        type=str,
        default="bart_gen_pr_description",
        help="Name of the output column for generated summaries (default: bart_gen_pr_description)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing. L4: 16-32, A100: 32-128 (default: 16)"
    )
    
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1024,
        help="Maximum input length for tokenization (default: 1024)"
    )
    
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=1000,
        help="Maximum output length for generation (default: 1000)"
    )
    
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable FP16 mixed precision (default: FP16 enabled for speed)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if using FP16
    use_fp16 = not args.no_fp16
    
    # Load model with FP16 if enabled
    load_model(str(model_path), use_fp16=use_fp16)
    
    # Load input data
    print(f"\nLoading data from: {input_path}")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix == ".json":
        df = pd.read_json(input_path)
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        print("Supported formats: .parquet, .csv, .json")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    
    # Process data with batch processing
    df_processed = process_dataframe(
        df, 
        column_name=args.column,
        output_col=args.output_column,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        use_fp16=use_fp16
    )
    
    # Save output
    print(f"\nSaving results to: {output_path}")
    if output_path.suffix == ".parquet":
        df_processed.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        df_processed.to_csv(output_path, index=False)
    elif output_path.suffix == ".json":
        df_processed.to_json(output_path, orient="records", indent=2)
    else:
        print(f"Error: Unsupported output format: {output_path.suffix}")
        print("Supported formats: .parquet, .csv, .json")
        sys.exit(1)
    
    print(f"✓ Processing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
