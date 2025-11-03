"""
Async job to fetch commit SHAs and repo_id from GitHub PRs.

This script:
1. Loads a parquet file with pr_id, pr_number, and repo columns
2. Calls GitHub API asynchronously to get PR details (repo_id) and commit SHAs for each PR
3. Returns a dataframe with pr_id, pr_number, repo, repo_id, and sha
4. Saves result to the same folder as input file
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# Configuration
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 30


async def fetch_pr_commits(
    session: aiohttp.ClientSession,
    pr_id: str,
    pr_number: int,
    repo: str,
    semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """
    Fetch all commit SHAs and repo_id for a single PR from GitHub API.
    
    Args:
        session: aiohttp client session
        pr_id: PR identifier
        pr_number: PR number
        repo: repository in format "owner/repo"
        semaphore: semaphore to limit concurrent requests
        
    Returns:
        List of dictionaries containing pr_id, pr_number, repo, repo_id, and sha
    """
    
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/commits"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                if response.status == 200:
                    commits = await response.json()
                    results = []
                    
                    for commit in commits:
                        results.append({
                            "pr_id": pr_id,
                            "pr_number": pr_number,
                            "repo": repo,
                            "sha": commit.get("sha")
                        })
                    
                    logger.info(f"Fetched {len(results)} commits for PR #{pr_number} in {repo}")
                    return results
                    
                elif response.status == 404:
                    logger.warning(f"PR not found: #{pr_number} in {repo}")
                    return []
                    
                elif response.status == 403:
                    logger.error(f"Rate limit exceeded or forbidden for PR #{pr_number}")
                    await asyncio.sleep(60)
                    return []
                    
                else:
                    logger.error(f"Failed to fetch PR #{pr_number}: HTTP {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching PR #{pr_number} in {repo}")
            return []
        except Exception as e:
            logger.error(f"Error fetching PR #{pr_number} in {repo}: {str(e)}")
            return []


async def fetch_all_pr_commits(
    df: pd.DataFrame,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
) -> List[Dict[str, Any]]:
    """
    Fetch commit SHAs for all PRs in the dataframe concurrently.
    
    Args:
        df: Input dataframe with columns: pr_id, pr_number, repo
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        List of all commit details
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in df.iterrows():
            task = fetch_pr_commits(
                session=session,
                pr_id=row["pr_id"],
                pr_number=int(row["pr_number"]),
                repo=row["repo"],
                semaphore=semaphore
            )
            tasks.append(task)
        
        logger.info(f"Starting {len(tasks)} async requests with max {max_concurrent} concurrent...")
        results = await asyncio.gather(*tasks)
    
    # Flatten the list of lists
    all_results = []
    for result in results:
        all_results.extend(result)
    
    return all_results


def load_and_fetch_pr_commits(
    input_path: str,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
) -> pd.DataFrame:
    """
    Main function to load PR data, fetch commit SHAs and repo_id, and save results.
    
    Args:
        input_path: Path to input parquet file with pr_id, pr_number, repo columns
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        DataFrame with pr_id, pr_number, repo, repo_id, sha columns
    """
    # Load input data
    logger.info(f"Loading input data from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Validate required columns
    required_columns = ["pr_id", "pr_number", "repo"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"Processing {len(df)} PRs from {df['repo'].nunique()} repositories")
    
    # Fetch commit SHAs asynchronously
    results = asyncio.run(fetch_all_pr_commits(df, max_concurrent))
    
    # Convert to dataframe
    result_df = pd.DataFrame(results)
    
    if len(result_df) > 0:
        logger.info(f"Fetched {len(result_df)} commit SHAs from {len(df)} PRs")
        
        # Save to same folder as input file
        input_path_obj = Path(input_path)
        output_filename = input_path_obj.stem + "_with_shas" + input_path_obj.suffix
        output_path = input_path_obj.parent / output_filename
        
        result_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Saved results to {output_path}")
    else:
        logger.warning("No results to save")
    
    return result_df


if __name__ == "__main__":
    # Load and process human PR data
    input_path = "/Users/dungp@backbase.com/Documents/aidev-mining/data/github/input/human_pr_repo.parquet"
    result_df = load_and_fetch_pr_commits(input_path, max_concurrent=10)