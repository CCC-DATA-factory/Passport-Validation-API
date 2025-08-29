# PowerShell script to initialize a git repository, create a .gitignore file, and push to GitHub

# create .gitignore using a here-string
@'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
pip-wheel-metadata/

# IDE/editor
.vscode/
.idea/
.DS_Store

# Logs & test outputs (do NOT push per-request logs)
tests/logs/
*.log

# Docker
*.tar
docker-compose.override.yml

# OS
Thumbs.db
'@ | Set-Content -Path .gitignore -Encoding UTF8

# initialize repo (if not already a git repo)
git init

# set main branch name
git branch -M main

# add the remote origin
git remote add origin https://github.com/CCC-DATA-factory/Passport-Validation-API.git

# configure user (if not set)
git config user.name "DevButterflies"
git config user.email "rabii.nasri1@gmail.com"

# add files and commit
git add .
git commit -m "Initial commit: Passport Validation API + stress tests + READMEs"

# push to remote (you will be prompted to authenticate)
git push -u origin main
