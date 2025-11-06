# Virtual environment setup
1. Install uv (example: on macOS)
```
brew install uv
```
2. Create and activate the virtual environment
```
uv venv --python 3.12 
source .venv/bin/activate
```
3. Install the dependencies
```
uv sync
```

# .env file
```
GITHUB_TOKEN_1=your_github_token_1
GITHUB_TOKEN_2=your_github_token_2
```


# Command to build human PR commit details dataframe
First time run:
```
python processing/build_human_pr_commit_details_df.py -o data/output.parquet
```
Resume from previous run:
```
python processing/build_human_pr_commit_details_df.py -o data/output.parquet --resume
```