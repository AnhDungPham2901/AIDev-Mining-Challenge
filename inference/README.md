# How to run on colab
1. Generate PR description using BART
```bash
python gen_pr_description_bart.py --input_path ./input/pr_df_commit_messages.csv --model_path /content/drive/MyDrive/colab/aidev-mining/bart-t5/results/model-trained-with-cleaned-dataset-BART/bart --output_path ./output/pr_df_pr_desc.parquet --column aggregated_commit_messages
```
2. Calculate BERTScore
```bash
python cal_bert_score.py --input ./input/input_to_bertscore.parquet --output ./output/body_ai_pr_desc_bertscore.parquet --cand-column pr_body --ref-column bart_gen_pr_description
```

3. Generate commit messages using CodeT5
```bash
python gen_commit_message.py --input_path ./input/commit_message_generation_df.parquet --output ./output/commit_message_gen_result.parquet
```