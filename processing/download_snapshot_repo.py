# at the time PR created 
import requests, os

def download_repo_snapshot(owner, repo, sha, save_dir):
    url = f"https://github.com/{owner}/{repo}/archive/{sha}.zip"
    r = requests.get(url)
    if r.status_code == 200:
        path = os.path.join(save_dir, f"{repo}-{sha}.zip")
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Saved {path}")
    else:
        print(f"Failed {url}: {r.status_code}")

# Example
download_repo_snapshot("pallets", "flask", "2f9d54dda4f0c87c19e0bbeb9936f525d0587e16", "./repos")
