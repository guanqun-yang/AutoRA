import base64
import requests
from datetime import datetime

REPO_OWNER = 'guanqun-yang'
REPO_NAME = 'literature-analytics'
LOCAL_FILE_PATH = 'test.md'
QUERY_NAME = "data poisoning"  # example search
COMMIT_MESSAGE = 'Upload search result and update README'

# Session setup
session = requests.Session()
session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
})

# Prepare date and paths
now = datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
timestamp = now.strftime("%Y%m%d")
safe_query_name = QUERY_NAME.replace(" ", "_").lower()
new_md_path = f"{year}/{month}/{timestamp}_{safe_query_name}.md"

# Upload the new search markdown
with open(LOCAL_FILE_PATH, 'rb') as f:
    content = f.read()

encoded_content = base64.b64encode(content).decode('utf-8')

upload_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{new_md_path}"

# Check if file already exists
r = session.get(upload_url)
if r.status_code == 200:
    sha = r.json()["sha"]
else:
    sha = None

payload = {
    "message": f"Upload {new_md_path}",
    "content": encoded_content,
    "branch": "main"
}
if sha:
    payload["sha"] = sha

r = session.put(upload_url, json=payload)
if r.status_code in [201, 200]:
    print(f"✅ Successfully uploaded {new_md_path}")
else:
    print(f"❌ Failed to upload {new_md_path}: {r.json()}")

# ---------------------------------------
# Now, update README.md
# ---------------------------------------

# 1. Get current README.md
readme_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/README.md"
r = session.get(readme_url)

if r.status_code == 200:
    readme_data = r.json()
    current_readme = base64.b64decode(readme_data['content']).decode('utf-8')
    readme_sha = readme_data['sha']
else:
    # No README exists yet
    current_readme = "# Search Results\n\n"
    readme_sha = None

# 2. Update README.md with a new link
new_link = f"- [{QUERY_NAME.title()} ({timestamp})]({new_md_path})\n"
updated_readme = current_readme + new_link

# 3. Upload new README.md
payload = {
    "message": "Update README.md with new search result",
    "content": base64.b64encode(updated_readme.encode('utf-8')).decode('utf-8'),
    "branch": "main"
}
if readme_sha:
    payload["sha"] = readme_sha

r = session.put(readme_url, json=payload)
if r.status_code in [201, 200]:
    print(f"✅ Successfully updated README.md")
else:
    print(f"❌ Failed to update README.md: {r.json()}")