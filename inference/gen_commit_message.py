#!/usr/bin/env python3
"""
CodeT5 Commit Message Generation Service (Optimized for NVIDIA GPUs)

This script loads the JetBrains CodeT5 model and generates commit messages from patch data.
Optimized for maximum speed with batch processing and mixed precision.
Supports A100, L4, and other NVIDIA GPUs.

Usage:
    python gen_commit_message.py --input_path <path> --output_path <path>
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


def load_model(use_fp16: bool = True):
    """Load the CodeT5 model and tokenizer from HuggingFace."""
    global model, tokenizer, device
    
    model_name = "JetBrains-Research/cmg-codet5-without-history"
    print(f"Loading model from HuggingFace: {model_name}")
    
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()  # Set to evaluation mode
    
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


def process_dataframe(df: pd.DataFrame, column_name: str = "patch", output_col: str = "message", 
                      batch_size: int = 16, max_input_length: int = 512, 
                      max_output_length: int = 128, use_fp16: bool = True):
    """Process a DataFrame with batch processing for maximum speed."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    print(f"Processing {len(df)} rows with batch size {batch_size}...")
    device_name = "CUDA" if device.type == "cuda" else "CPU"
    
    # Process in batches
    results = []
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(df), batch_size), desc=f"Generating Commit Messages ({device_name}, FP16={use_fp16})", total=num_batches):
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
    print(f"\n✓ Generated {generated_count}/{len(df)} commit messages successfully")
    
    return df


def main():
    """Main function to run the service."""
    parser = argparse.ArgumentParser(
        description="Generate commit messages using CodeT5 model (Optimized for NVIDIA GPUs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python gen_commit_message.py --input_path data/input.parquet --output_path data/output.parquet
  
  # Short form
  python gen_commit_message.py -i input.parquet -o output.parquet
  
  # Custom batch size
  python gen_commit_message.py -i input.parquet -o output.parquet --batch_size 32
        """
    )
    
    parser.add_argument(
        "--input_path", "-i",
        type=str,
        required=True,
        help="Path to input parquet file (must contain 'patch' column)"
    )
    
    parser.add_argument(
        "--output_path", "-o",
        type=str,
        required=True,
        help="Path to save output parquet file (will contain 'message' column)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing. L4: 16-32, A100: 32-128 (default: 16)"
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
    
    if input_path.suffix != ".parquet":
        print(f"Error: Input file must be a parquet file, got: {input_path.suffix}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix != ".parquet":
        print(f"Error: Output file must be a parquet file, got: {output_path.suffix}")
        sys.exit(1)
    
    # Determine if using FP16
    use_fp16 = not args.no_fp16
    
    # Load model with FP16 if enabled
    load_model(use_fp16=use_fp16)
    
    # Load input data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    
    # Verify 'patch' column exists
    if 'patch' not in df.columns:
        print(f"Error: Input dataframe must contain 'patch' column")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Process data with batch processing
    df_processed = process_dataframe(
        df, 
        column_name="patch",
        output_col="message",
        batch_size=args.batch_size,
        max_input_length=512,
        max_output_length=128,
        use_fp16=use_fp16
    )
    
    # Save output
    print(f"\nSaving results to: {output_path}")
    df_processed.to_parquet(output_path, index=False)
    
    print(f"✓ Processing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
