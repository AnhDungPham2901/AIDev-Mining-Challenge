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

# Code structure
- `notebooks/`: This folder contains the main code for this research project
- `processing/`: This folder contains the utils for processing the data and the code to build the human PR commit details dataframe
- `inference/`: This folder contains the code we use to run the inference on Google Colab
- `data/`: This folder contains the data of the processing steps. Find this folder
- `plots/`: This folder contains the plots of the analysis results


# Command to build human PR commit details dataframe
First time run:
```
python processing/build_human_pr_commit_details_df.py -o data/output.parquet
```
Resume from previous run:
```
python processing/build_human_pr_commit_details_df.py -o data/output.parquet --resume
```

# Code and Data
We uploaded the data files and the whole code to Google Drive: https://drive.google.com/file/d/1vgMOl1R3oNwn3yuJIvVrzKzDA9NGfBIH/view?usp=drive_link

# Google Colab
The inference for commit message generation, and generate embeddings are run on Google Colab. You can find the code and the output of the runs here: https://drive.google.com/drive/folders/1tD_-t9q-3bhLX0-3suuxojvbgucMAHGD?usp=sharing