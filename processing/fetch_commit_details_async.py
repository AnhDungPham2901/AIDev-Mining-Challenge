"""
Async job to fetch commit details from GitHub API.

This script:
1. Loads an input dataframe with sha, repo, and pr_id columns
2. Calls GitHub API asynchronously to get commit details
3. Returns a table with sha, pr_id, file as keys
4. Stores output as a parquet file
"""

import asyncio
import os
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
MAX_CONCURRENT_REQUESTS = 10  # Limit concurrent requests to avoid rate limiting
REQUEST_TIMEOUT = 30  # Timeout in seconds


async def fetch_commit_detail(
    session: aiohttp.ClientSession,
    sha: str,
    repo: str,
    pr_id: str,
    repo_id: str,
    semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """
    Fetch commit details for a single commit from GitHub API.
    
    Args:
        session: aiohttp client session
        sha: commit SHA
        repo: repository in format "owner/repo"
        pr_id: pull request ID
        semaphore: semaphore to limit concurrent requests
        
    Returns:
        List of dictionaries containing file-level commit details
    """
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Extract file-level details
                    for file_data in data.get("files", []):
                        results.append({
                            "sha": sha,
                            "pr_id": pr_id,
                            "repo_id": repo_id,
                            "file": file_data.get("filename"),
                            "status": file_data.get("status"),
                            "additions": file_data.get("additions"),
                            "deletions": file_data.get("deletions"),
                            "changes": file_data.get("changes"),
                            "patch": file_data.get("patch"),
                            "message": data.get("commit", {}).get("message"),
                            "author": data.get("commit", {}).get("author", {}).get("name"),
                            "author_email": data.get("commit", {}).get("author", {}).get("email"),
                            "date": data.get("commit", {}).get("author", {}).get("date"),
                        })
                    
                    logger.info(f"Fetched {len(results)} files for commit {sha[:7]}")
                    return results
                    
                elif response.status == 404:
                    logger.warning(f"✗ Commit not found: {sha[:7]} in {repo}")
                    return []
                    
                elif response.status == 403:
                    logger.error(f"✗ Rate limit exceeded or forbidden: {sha[:7]}")
                    # Wait a bit before continuing
                    await asyncio.sleep(60)
                    return []
                    
                else:
                    logger.error(f"✗ Failed to fetch {sha[:7]}: HTTP {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"✗ Timeout fetching commit {sha[:7]}")
            return []
        except Exception as e:
            logger.error(f"✗ Error fetching commit {sha[:7]}: {str(e)}")
            return []


async def fetch_all_commits(
    df: pd.DataFrame,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
) -> List[Dict[str, Any]]:
    """
    Fetch commit details for all rows in the dataframe concurrently.
    
    Args:
        df: Input dataframe with columns: sha, repo, pr_id
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        List of all file-level commit details
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in df.iterrows():
            task = fetch_commit_detail(
                session=session,
                sha=row["sha"],
                repo=row["repo"],
                pr_id=row.get("pr_id"),
                repo_id=row.get("repo_id"),
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


def load_and_fetch_commits(
    input_path: str,
    output_path: str,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
) -> pd.DataFrame:
    """
    Main function to load input data, fetch commit details, and save to parquet.
    
    Args:
        input_path: Path to input parquet/csv file with sha, repo, pr_id columns
        output_path: Path to output parquet file
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        DataFrame with commit details
    """
    # Load input data
    logger.info(f"Loading input data from {input_path}")
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input file must be .parquet or .csv")
    
    # Validate required columns
    required_columns = ["sha", "repo"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Add pr_id if not present
    if "pr_id" not in df.columns:
        logger.error("pr_id column not found, using empty string")
        raise ValueError("pr_id column not found")
    
    logger.info(f"Processing {len(df)} commits from {df['repo'].nunique()} repositories")
    
    # Fetch commit details asynchronously
    results = asyncio.run(fetch_all_commits(df, max_concurrent))
    
    # Convert to dataframe
    result_df = pd.DataFrame(results)
    
    if len(result_df) > 0:
        logger.info(f"Fetched details for {len(result_df)} files")
        
        # Save to parquet
        result_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Saved results to {output_path}")
    else:
        logger.warning("No results to save")
    
    return result_df


if __name__ == "__main__":
    # example with 1 fetch_commit_detail
    input_path = "/Users/dungp@backbase.com/Documents/aidev-mining/data/test_df.parquet"
    output_path = "/Users/dungp@backbase.com/Documents/aidev-mining/data/test_df_with_file_level_details.parquet"
    max_concurrent = 10
    result_df = load_and_fetch_commits(input_path, output_path, max_concurrent)
    print(result_df.head())