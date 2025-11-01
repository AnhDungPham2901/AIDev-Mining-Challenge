import requests
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

OWNER = "getsentry"
REPO = "sentry"
PR_NUMBER = 89131


headers = {"Authorization": f"token {GITHUB_TOKEN}"}

url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}/commits"
response = requests.get(url, headers=headers)
response.raise_for_status()

commits = response.json()
shas = [commit["sha"] for commit in commits]
print("SHAs in PR:", shas)