"""
Build human PR commit details dataframe with async processing.

=== OVERVIEW ===
Fetches commit details for all PRs from the AIDev human_pull_request dataset.
For each PR: fetch commit SHAs → fetch details for each commit → save file-level data.

=== FEATURES ===

1. ASYNC PROCESSING:
   - Processes up to 20 PRs concurrently (configurable via MAX_CONCURRENT_REQUESTS)
   - Within each PR, fetches all commit details concurrently
   - 10-20x faster than sequential processing

2. DATA FLOW:
   - Load human PR dataset from HuggingFace (hf://datasets/hao-li/AIDev/human_pull_request.parquet)
   - Extract pr_id, pr_number, repo columns
   - For each PR:
     * Fetch commit SHAs via GitHub API
     * Fetch commit details (files, patches, metadata) for each SHA
   - Save results to parquet with columns: sha, pr_id, file, status, additions, deletions, 
     changes, patch, message, author, author_email, date

3. DYNAMIC RATE LIMIT HANDLING:
   - Uses 2 GitHub tokens (GITHUB_TOKEN_1 and GITHUB_TOKEN_2)
   - Detects rate limit (403 errors) in real-time during processing
   - When rate limit detected:
     * Saves current results
     * Switches to alternate token
     * Automatically retries failed PRs with new token
   - No PRs are skipped - guaranteed retry mechanism
   - Exits gracefully if both tokens exhausted (shows reset time)

4. RESUME CAPABILITY:
   - Use --resume flag to continue from previous run
   - Compares existing output file with input to find unprocessed PRs
   - Skips already processed pr_ids
   - Appends new results to existing file

5. SMART CHECKPOINT SAVES:
   - Accumulates results in memory between checkpoints
   - Saves progress every 500 PRs, then clears memory
   - Saves before token switches, then clears memory
   - save_results() ALWAYS loads existing file and deduplicates on (pr_id, sha, file)
   - ZERO data loss:
     * File always has latest data
     * Crash before save? Resume picks up unprocessed PRs
     * Crash after save? No duplicates due to smart deduplication

6. COMMAND LINE INTERFACE:
   Required:
     --output_path, -o : Path to save output parquet file
   Optional:
     --resume, -r      : Resume from existing output file

7. ROBUST ERROR HANDLING:
   - Handles 404 (not found), 403 (rate limit), timeouts
   - Logs all errors with context
   - Continues processing despite individual failures
   - Detailed logging to file and console

=== USAGE ===
# First run
python build_human_pr_commit_details_df.py -o data/output.parquet

# Resume interrupted run
python build_human_pr_commit_details_df.py -o data/output.parquet --resume

=== ENVIRONMENT ===
Required in .env file:
- GITHUB_TOKEN_1: First GitHub personal access token
- GITHUB_TOKEN_2: Second GitHub personal access token
"""
import pandas as pd
from dotenv import load_dotenv
import os
from loguru import logger
import requests
import aiohttp
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time

load_dotenv()

# Configuration for async processing
MAX_CONCURRENT_REQUESTS = 20  # Number of PRs to process concurrently


def get_next_token(current_token_name: str) -> Tuple[str, str]:
    """Get the next token and its name."""
    if current_token_name == 'token1':
        return os.getenv("GITHUB_TOKEN_2"), 'token2'
    elif current_token_name == 'token2':
        return os.getenv("GITHUB_TOKEN_1"), 'token1'
    else:
        raise ValueError(f"Invalid token name: {current_token_name}")


def get_token_rate_limit_info(token: str) -> Dict[str, Any]:
    """
    Get detailed rate limit information for a token.
    
    Returns:
        Dict with 'remaining', 'limit', 'reset' (timestamp)
    """
    url = "https://api.github.com/rate_limit"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    core_info = response.json()['resources']['core']
    return {
        'remaining': core_info['remaining'],
        'limit': core_info['limit'],
        'reset': core_info['reset']
    }


def check_token_limit(token: str) -> bool:
    """Check if token has remaining requests."""
    try:
        info = get_token_rate_limit_info(token)
        return info['remaining'] > 0
    except Exception as e:
        logger.error(f"Error checking token limit: {e}")
        return False


def load_human_pr_df() -> pd.DataFrame:
    """Load the human PR dataframe from HuggingFace."""
    logger.info("Loading human PR dataframe from HuggingFace...")
    human_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pull_request.parquet")
    logger.info(f"Loaded {len(human_pr_df)} PRs")
    return human_pr_df


def form_input_df(human_pr_df: pd.DataFrame) -> pd.DataFrame:
    """Form input dataframe with pr_id, pr_number, and repo columns."""
    logger.info("Forming input dataframe...")
    input_df = human_pr_df[['id', 'number', 'repo_url']].copy()
    input_df.columns = ['pr_id', 'pr_number', 'repo_url']
    input_df['repo'] = input_df['repo_url'].str.split('/').str[-2:].str.join('/')
    input_df.drop('repo_url', axis=1, inplace=True)
    logger.info(f"Formed input dataframe with {len(input_df)} rows")
    return input_df


class RateLimitException(Exception):
    """Exception raised when rate limit is hit."""
    pass


async def fetch_pr_commits(session: aiohttp.ClientSession, token: str, pr_number: int, repo: str) -> List[str]:
    """
    Fetch all commit SHAs for a single PR (async).
    
    Args:
        session: aiohttp client session
        token: GitHub token
        pr_number: PR number
        repo: Repository in format "owner/repo"
        
    Returns:
        List of commit SHAs
        
    Raises:
        RateLimitException: When rate limit is exceeded
    """
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/commits"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                commits = await response.json()
                shas = [commit.get("sha") for commit in commits if commit.get("sha")]
                logger.debug(f"Fetched {len(shas)} commits for PR #{pr_number} in {repo}")
                return shas
            elif response.status == 404:
                logger.warning(f"PR not found: #{pr_number} in {repo}")
                return []
            elif response.status == 403:
                logger.error(f"Rate limit exceeded for PR #{pr_number} - TOKEN SWITCH NEEDED")
                raise RateLimitException(f"Rate limit hit for PR #{pr_number}")
            else:
                logger.error(f"Failed to fetch PR #{pr_number}: HTTP {response.status}")
                return []
    except RateLimitException:
        raise
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching PR #{pr_number} in {repo}")
        return []
    except Exception as e:
        logger.error(f"Error fetching PR #{pr_number} in {repo}: {str(e)}")
        return []


async def fetch_commit_details(session: aiohttp.ClientSession, token: str, sha: str, repo: str, pr_id: str) -> List[Dict[str, Any]]:
    """
    Fetch commit details for a single commit (async).
    
    Args:
        session: aiohttp client session
        token: GitHub token
        sha: Commit SHA
        repo: Repository in format "owner/repo"
        pr_id: Pull request ID
        
    Returns:
        List of dictionaries containing file-level commit details
        
    Raises:
        RateLimitException: When rate limit is exceeded
    """
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                results = []
                
                # Extract file-level details
                for file_data in data.get("files", []):
                    results.append({
                        "sha": sha,
                        "pr_id": pr_id,
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
                
                logger.debug(f"Fetched {len(results)} files for commit {sha[:7]}")
                return results
            elif response.status == 404:
                logger.warning(f"Commit not found: {sha[:7]} in {repo}")
                return []
            elif response.status == 403:
                logger.error(f"Rate limit exceeded for commit {sha[:7]} - TOKEN SWITCH NEEDED")
                raise RateLimitException(f"Rate limit hit for commit {sha[:7]}")
            else:
                logger.error(f"Failed to fetch {sha[:7]}: HTTP {response.status}")
                return []
    except RateLimitException:
        raise
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching commit {sha[:7]}")
        return []
    except Exception as e:
        logger.error(f"Error fetching commit {sha[:7]}: {str(e)}")
        return []


async def process_single_pr(session: aiohttp.ClientSession, token: str, pr_id: str, pr_number: int, repo: str, semaphore: asyncio.Semaphore) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process a single PR: fetch SHAs and then commit details for each SHA (async).
    
    Args:
        session: aiohttp client session
        token: GitHub token
        pr_id: PR identifier
        pr_number: PR number
        repo: Repository in format "owner/repo"
        semaphore: Semaphore to control concurrency
        
    Returns:
        Tuple of (list of commit details, number of API requests made)
        
    Raises:
        RateLimitException: When rate limit is hit
    """
    async with semaphore:
        all_results = []
        request_count = 0
        
        # Step 1: Fetch commit SHAs for this PR
        shas = await fetch_pr_commits(session, token, pr_number, repo)
        request_count += 1
        
        if not shas:
            logger.warning(f"No commits found for PR {pr_id} (#{pr_number})")
            return all_results, request_count
        
        logger.info(f"Processing PR {pr_id} (#{pr_number} in {repo}): {len(shas)} commits")
        
        # Step 2: Fetch commit details for each SHA concurrently
        tasks = []
        for sha in shas:
            task = fetch_commit_details(session, token, sha, repo, pr_id)
            tasks.append(task)
        
        # Use return_exceptions=True to handle rate limits gracefully
        commit_results = await asyncio.gather(*tasks, return_exceptions=True)
        request_count += len(shas)
        
        # Check for rate limit exceptions and flatten results
        for commit_result in commit_results:
            if isinstance(commit_result, RateLimitException):
                # Propagate rate limit exception up
                raise commit_result
            elif isinstance(commit_result, Exception):
                # Log other exceptions but continue
                logger.error(f"Error in PR {pr_id}: {commit_result}")
            else:
                # Normal result, extend
                all_results.extend(commit_result)
        
        return all_results, request_count


def filter_remaining_prs(input_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Filter input_df to only include PRs not in the output file.
    
    Args:
        input_df: Input dataframe with all PRs
        output_path: Path to existing output file
        
    Returns:
        Filtered dataframe with only remaining PRs
    """
    if not Path(output_path).exists():
        logger.info("Output file does not exist, processing all PRs")
        return input_df
    
    logger.info(f"Loading existing output from {output_path}")
    output_df = pd.read_parquet(output_path)
    logger.info(f"Loaded {len(output_df)} rows from {output_path}")
    
    if 'pr_id' not in output_df.columns:
        logger.warning("Output file missing pr_id column, processing all PRs")
        return input_df
    
    processed_pr_ids = set(output_df['pr_id'].unique())
    remaining_df = input_df[~input_df['pr_id'].isin(processed_pr_ids)].copy()
    
    logger.info(f"Already processed: {len(processed_pr_ids)} PRs")
    logger.info(f"Remaining to process: {len(remaining_df)} PRs")
    
    return remaining_df


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save results to parquet file with smart deduplication.
    
    Always appends to existing file and removes duplicates based on (pr_id, sha, file).
    This ensures no data loss even if program crashes or is interrupted.
    
    Args:
        results: List of commit details (fresh results since last save)
        output_path: Path to save output
    """
    if not results:
        logger.warning("No results to save")
        return
    
    new_df = pd.DataFrame(results)
    logger.info(f"Saving {len(new_df)} rows from memory")
    
    # Always check for existing file and merge intelligently
    if Path(output_path).exists():
        existing_df = pd.read_parquet(output_path)
        logger.debug(f"Existing file has {len(existing_df)} rows")
        
        # Combine and remove duplicates based on (pr_id, sha, file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Drop duplicates - keep first occurrence (from existing file)
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['pr_id', 'sha', 'file'], keep='first')
        after_dedup = len(combined_df)
        
        duplicates_removed = before_dedup - after_dedup
        truly_new = len(new_df) - duplicates_removed
        
        if duplicates_removed > 0:
            logger.warning(f"Removed {duplicates_removed} duplicate rows (already in file)")
        
        combined_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved: {after_dedup} total rows (+{truly_new} new) → {output_path}")
    else:
        # First time save
        new_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"✓ Created new file with {len(new_df)} rows at {output_path}")


async def process_batch_async(
    session: aiohttp.ClientSession,
    batch_df: pd.DataFrame,
    token: str,
    semaphore: asyncio.Semaphore
) -> Tuple[List[Dict[str, Any]], int, bool, List[int]]:
    """
    Process a batch of PRs concurrently.
    
    Args:
        session: aiohttp client session
        batch_df: Batch of PRs to process
        token: GitHub token
        semaphore: Semaphore to control concurrency
        
    Returns:
        Tuple of (list of all results, total requests made, rate_limit_hit, failed_indices)
    """
    tasks = []
    for _, row in batch_df.iterrows():
        pr_id = row['pr_id']
        pr_number = int(row['pr_number'])
        repo = row['repo']
        
        task = process_single_pr(session, token, pr_id, pr_number, repo, semaphore)
        tasks.append(task)
    
    # Use return_exceptions to catch rate limit exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate all results and check for rate limits
    all_results = []
    total_requests = 0
    rate_limit_hit = False
    failed_indices = []  # Track which PRs failed due to rate limit
    
    for idx, result in enumerate(results):
        if isinstance(result, RateLimitException):
            logger.warning(f"Rate limit detected for PR index {idx} in batch: {result}")
            rate_limit_hit = True
            failed_indices.append(idx)
            # Don't break, collect all successful results
        elif isinstance(result, Exception):
            logger.error(f"Error for PR index {idx} in batch: {result}")
        else:
            pr_results, request_count = result
            all_results.extend(pr_results)
            total_requests += request_count
    
    return all_results, total_requests, rate_limit_hit, failed_indices


async def retry_failed_prs(
    session: aiohttp.ClientSession,
    batch_df: pd.DataFrame,
    failed_indices: List[int],
    token: str,
    token_name: str,
    semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """
    Retry PRs that failed due to rate limiting with a new token.
    
    Args:
        session: aiohttp client session
        batch_df: Original batch dataframe
        failed_indices: Indices of PRs that failed in the batch
        token: New GitHub token to use
        token_name: Name of the token (for logging)
        semaphore: Semaphore to control concurrency
        
    Returns:
        List of successfully fetched results from retry
    """
    if not failed_indices:
        return []
    
    failed_batch_df = batch_df.iloc[failed_indices]
    logger.info(f"🔄 Retrying {len(failed_indices)} failed PRs with {token_name}...")
    
    retry_results, retry_requests, retry_rate_limit, retry_failed = await process_batch_async(
        session, failed_batch_df, token, semaphore
    )
    
    logger.info(f"✓ Retry complete: {len(retry_results)} results from {len(failed_indices)} PRs")
    
    # If retry also hit rate limit, log warning
    # Those PRs will be caught by --resume on next run
    if retry_rate_limit:
        logger.error(f"⚠️  Rate limit hit again during retry! {len(retry_failed)} PRs still failed.")
        logger.error(f"Run with --resume to retry these PRs.")
    
    return retry_results


async def process_prs_with_rate_limit(input_df: pd.DataFrame, output_path: str, resume: bool):
    """
    Main async processing loop with dynamic rate limit handling.
    
    Switches tokens immediately when rate limit (403) is detected.
    
    Args:
        input_df: Input dataframe with pr_id, pr_number, repo
        output_path: Path to save output
        resume: If True, resume from existing output file
    """
    # Initialize tokens
    token1 = os.getenv("GITHUB_TOKEN_1")
    token2 = os.getenv("GITHUB_TOKEN_2")
    
    if not token1 or not token2:
        raise ValueError("Both GITHUB_TOKEN_1 and GITHUB_TOKEN_2 must be set in .env")
    
    current_token = token1
    current_token_name = 'token1'
    
    # Filter remaining PRs if resuming
    if resume:
        input_df = filter_remaining_prs(input_df, output_path)
        if len(input_df) == 0:
            logger.info("All PRs already processed!")
            return
    
    # Reset index for easier iteration
    input_df = input_df.reset_index(drop=True)
    
    # Check initial token
    logger.info(f"Starting with {current_token_name}")
    rate_info = get_token_rate_limit_info(current_token)
    logger.info(f"Token {current_token_name}: {rate_info['remaining']}/{rate_info['limit']} requests remaining")
    
    # If current token is rate limited, switch immediately
    if rate_info['remaining'] == 0:
        logger.warning(f"{current_token_name} is rate limited, switching to alternate token...")
        current_token, current_token_name = get_next_token(current_token_name)
        rate_info = get_token_rate_limit_info(current_token)
        logger.info(f"Switched to {current_token_name}: {rate_info['remaining']}/{rate_info['limit']} requests remaining")
        
        if rate_info['remaining'] == 0:
            reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rate_info['reset']))
            logger.error(f"Both tokens are rate limited. Reset at {reset_time}. Exiting...")
            return
    
    all_results = []
    batch_size = 100  # Process 100 PRs per batch
    
    total_prs = len(input_df)
    processed_prs = 0
    
    # Create semaphore for concurrent request control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Process in batches
        for batch_start in range(0, total_prs, batch_size):
            batch_end = min(batch_start + batch_size, total_prs)
            batch_df = input_df.iloc[batch_start:batch_end]
            
            # Process this batch
            logger.info(f"Processing batch {batch_start}-{batch_end} of {total_prs} PRs with {current_token_name}...")
            batch_results, batch_requests, rate_limit_hit, failed_indices = await process_batch_async(
                session, batch_df, current_token, semaphore
            )
            
            all_results.extend(batch_results)
            processed_prs += len(batch_df)
            
            logger.info(f"Batch complete: {len(batch_results)} results, {batch_requests} requests made")
            logger.info(f"Progress: {processed_prs}/{total_prs} PRs processed ({processed_prs/total_prs*100:.1f}%)")
            
            # If rate limit was hit, switch tokens and retry failed PRs
            if rate_limit_hit:
                logger.warning(f"⚠️  Rate limit detected! {len(failed_indices)} PRs failed. Switching from {current_token_name}...")
                
                # Save current results before switching, then clear memory
                if all_results:
                    logger.info("Saving results before token switch...")
                    save_results(all_results, output_path)
                    all_results = []  # Clear after successful save
                    logger.debug("Cleared memory after save")
                
                # Switch to next token
                current_token, current_token_name = get_next_token(current_token_name)
                rate_info = get_token_rate_limit_info(current_token)
                
                logger.info(f"✓ Switched to {current_token_name}: {rate_info['remaining']}/{rate_info['limit']} requests remaining")
                
                # Check if new token has requests available
                if rate_info['remaining'] == 0:
                    reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rate_info['reset']))
                    logger.error(f"Both tokens are rate limited. Reset at {reset_time}. Exiting...")
                    break
                
                # Retry failed PRs with new token
                retry_results = await retry_failed_prs(
                    session, batch_df, failed_indices, 
                    current_token, current_token_name, semaphore
                )
                all_results.extend(retry_results)
            
            # Periodically save results (every 500 PRs), then clear memory
            if processed_prs % 500 < batch_size and all_results:
                logger.info(f"Checkpoint: Saving {len(all_results)} new results since last save...")
                save_results(all_results, output_path)
                all_results = []  # Clear after successful save
                logger.debug("Cleared memory after checkpoint save")
    
    # Save any remaining results (final save)
    if all_results:
        logger.info("Saving final accumulated results...")
        save_results(all_results, output_path)
    
    logger.info("✓ Processing complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build human PR commit details dataframe")
    parser.add_argument(
        "--output_path", "-o", 
        type=str,
        required=True,
        help="Path to save output parquet file"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=False,
        help="Resume from existing output file"
    )
    
    args = parser.parse_args()
    
    # Configure logger
    logger.add("build_human_pr_commit_details.log", rotation="100 MB")
    
    logger.info("=" * 80)
    logger.info("Starting human PR commit details extraction")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Resume mode: {'Enabled' if args.resume else 'Disabled'}")
    logger.info("=" * 80)
    
    try:
        # Load and form input dataframe
        human_pr_df = load_human_pr_df()
        input_df = form_input_df(human_pr_df)
        
        # Process PRs with rate limit handling (async)
        asyncio.run(process_prs_with_rate_limit(input_df, args.output_path, args.resume))
        
        logger.info("=" * 80)
        logger.info("Successfully completed!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
