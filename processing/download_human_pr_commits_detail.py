import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# Load your commit details
df = pd.read_csv("pr_commit_details.csv")

missing = df[df["patch"].isna()]  # only fetch where patch missing

results = []

for _, row in missing.iterrows():
    sha = row["sha"]
    repo = "owner/repo"  # TODO: replace with correct repo or add repo column
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        for f in data.get("files", []):
            results.append({
                "sha": sha,
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "changes": f.get("changes"),
                "patch": f.get("patch"),
                "message": data.get("commit", {}).get("message")
            })
    else:
        print(f"Failed {sha}: {r.status_code}")
    sleep(1)  # avoid GitHub rate limit

pd.DataFrame(results).to_csv("file_level_patches.csv", index=False)
print("Saved file_level_patches.csv ✅")
