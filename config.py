# config path for the project using relative path from the root of the project
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GITHUB_DATA_DIR = os.path.join(DATA_DIR, "github")

# Basic
PR_DF_PATH = "hf://datasets/hao-li/AIDev/pull_request.parquet"
REPO_DF_PATH = "hf://datasets/hao-li/AIDev/repository.parquet"
USER_DF_PATH = "hf://datasets/hao-li/AIDev/user.parquet"

# Comments and reviews
PR_COMMENTS_DF_PATH = "hf://datasets/hao-li/AIDev/pr_comments.parquet"
PR_REVIEWS_DF_PATH = "hf://datasets/hao-li/AIDev/pr_reviews.parquet"
PR_REVIEW_COMMENTS_DF_PATH = "hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet"

# Commits
PR_COMMITS_DF_PATH = "hf://datasets/hao-li/AIDev/pr_commits.parquet"
PR_COMMIT_DETAILS_DF_PATH = "hf://datasets/hao-li/AIDev/pr_commit_details.parquet"

# Related issues
RELATED_ISSUE_DF_PATH = "hf://datasets/hao-li/AIDev/related_issue.parquet"
ISSUE_DF_PATH = "hf://datasets/hao-li/AIDev/issue.parquet"

# Events
PR_TIMELINE_DF_PATH = "hf://datasets/hao-li/AIDev/pr_timeline.parquet"

# Task type
PR_TASK_TYPE_DF_PATH = "hf://datasets/hao-li/AIDev/pr_task_type.parquet"

# Human-PR
HUMAN_PR_DF_PATH = "hf://datasets/hao-li/AIDev/human_pull_request.parquet"
HUMAN_PR_TASK_TYPE_DF_PATH = "hf://datasets/hao-li/AIDev/human_pr_task_type.parquet"
HUMAN_PR_COMMIT_DETAILS_DF_PATH = os.path.join(GITHUB_DATA_DIR, "output", "human_pr_commit_details_df.parquet")


# Analysis files
ANALYSIS_1_1_DF_PATH = os.path.join(DATA_DIR, "analysis", "analysis_1_1_df.parquet")
ANALYSIS_1_2_1_DF_PATH = os.path.join(DATA_DIR, "analysis", "analysis_1_2_1_df.parquet")
ANALYSIS_1_2_2_DF_PATH = os.path.join(DATA_DIR, "analysis", "analysis_1_2_2_df.parquet") 