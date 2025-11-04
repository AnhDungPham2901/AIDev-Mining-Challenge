import argparse
import pandas as pd
import torch
from bert_score import score
from pathlib import Path
from tqdm import tqdm


def calculate_bert_score(
    input_path: str,
    output_path: str,
    cand_column: str = "generated_description",
    ref_column: str = "pr_description",
    batch_size: int = 32,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    rescale_with_baseline: bool = True,
):
    """
    Calculate BERTScore for candidate-reference pairs from input file.
    
    Args:
        input_path: Path to input parquet/csv file
        output_path: Path to save output parquet file with scores
        cand_column: Column name containing candidate texts
        ref_column: Column name containing reference texts
        batch_size: Batch size for processing
        model_type: BERTScore model type
        lang: Language code
        rescale_with_baseline: Whether to rescale with baseline
    """
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load input data
    input_path = Path(input_path)
    print(f"\nLoading data from: {input_path}")
    
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .parquet or .csv")
    
    print(f"Loaded {len(df)} rows")
    
    # Validate required columns
    if cand_column not in df.columns:
        raise ValueError(f"Column '{cand_column}' not found in input file. Available columns: {df.columns.tolist()}")
    if ref_column not in df.columns:
        raise ValueError(f"Column '{ref_column}' not found in input file. Available columns: {df.columns.tolist()}")
    
    # Filter out rows with missing values
    df_clean = df[[cand_column, ref_column]].dropna()
    print(f"Processing {len(df_clean)} rows (dropped {len(df) - len(df_clean)} rows with missing values)")
    
    candidates = df_clean[cand_column].tolist()
    references = df_clean[ref_column].tolist()
    
    # Calculate BERTScore in batches for better memory management
    print(f"\nCalculating BERTScore with model: {model_type}")
    print(f"Batch size: {batch_size}")
    
    all_precision = []
    all_recall = []
    all_f1 = []
    
    num_batches = (len(candidates) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(candidates), batch_size), total=num_batches, desc="Processing batches"):
        batch_cands = candidates[i:i + batch_size]
        batch_refs = references[i:i + batch_size]
        
        # Calculate scores for this batch
        P, R, F1 = score(
            batch_cands,
            batch_refs,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            model_type=model_type,
            device=device,
            verbose=False,
        )
        
        all_precision.extend(P.cpu().tolist())
        all_recall.extend(R.cpu().tolist())
        all_f1.extend(F1.cpu().tolist())
        
        # Clear cache to prevent memory issues
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Create results dataframe
    results_df = df_clean.copy()
    results_df["bert_precision"] = all_precision
    results_df["bert_recall"] = all_recall
    results_df["bert_f1"] = all_f1
    
    # Merge back with original dataframe to preserve all columns
    df_with_scores = df.merge(
        results_df[[cand_column, ref_column, "bert_precision", "bert_recall", "bert_f1"]],
        on=[cand_column, ref_column],
        how="left"
    )
    
    # Print summary statistics
    print("\n" + "="*50)
    print("BERTScore Summary Statistics")
    print("="*50)
    print(f"Precision: {pd.Series(all_precision).mean():.4f} ± {pd.Series(all_precision).std():.4f}")
    print(f"Recall:    {pd.Series(all_recall).mean():.4f} ± {pd.Series(all_recall).std():.4f}")
    print(f"F1:        {pd.Series(all_f1).mean():.4f} ± {pd.Series(all_f1).std():.4f}")
    print("="*50)
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_path}")
    df_with_scores.to_parquet(output_path, index=False)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BERTScore for candidate-reference pairs with CUDA support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input parquet or csv file",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output parquet file",
    )
    
    parser.add_argument(
        "--cand-column",
        type=str,
        default="generated_description",
        help="Column name containing candidate texts",
    )
    
    parser.add_argument(
        "--ref-column",
        type=str,
        default="pr_description",
        help="Column name containing reference texts",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore model type",
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code",
    )
    
    parser.add_argument(
        "--no-rescale",
        action="store_true",
        help="Disable rescaling with baseline",
    )
    
    args = parser.parse_args()
    
    calculate_bert_score(
        input_path=args.input,
        output_path=args.output,
        cand_column=args.cand_column,
        ref_column=args.ref_column,
        batch_size=args.batch_size,
        model_type=args.model_type,
        lang=args.lang,
        rescale_with_baseline=not args.no_rescale,
    )


if __name__ == "__main__":
    main()

