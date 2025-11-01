import requests
import pandas as pd
from time import sleep
import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.diff"}
headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.diff"}


repo = "AgentOps-AI/agentops"
sha = "2f9d54dda4f0c87c19e0bbeb9936f525d0587e16"

url = f"https://api.github.com/repos/{repo}/commits/{sha}"
r = requests.get(url, headers=headers)
if r.status_code == 200:
    with open(f"{sha}.patch", "w") as f:
        f.write(r.text)
else:
    print(f"Failed {sha}: {r.status_code}")
