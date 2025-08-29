<#
push.ps1

A safer, simpler PowerShell helper to:
- write a sensible .gitignore
- initialize a git repo if needed
- configure remote origin
- commit local changes
- fetch remote and attempt to integrate remote main
- push, with fallbacks (rebase -> merge -> force-with-lease)

Notes:
- This script prefers a non-destructive workflow; it will attempt `git pull --rebase` first.
- If rebase fails, it will abort the rebase and attempt a merge that prefers remote changes on conflict.
- As a last resort it will attempt `git push --force-with-lease`.
- Inspect output carefully before accepting force pushes.
#>

# Stop on first error
$ErrorActionPreference = 'Stop'

# ----------------- Helpers -----------------
function Run-Git {
    param(
        [Parameter(Mandatory=$true)][string[]]$Args
    )
    $cmd = "git " + ($Args -join ' ')
    Write-Host "`n> $cmd"
    $output = & git @Args 2>&1
    $exit = $LASTEXITCODE
    if ($output) { $output | ForEach-Object { Write-Host $_ } }
    return @{ ExitCode = $exit; Output = $output -join "`n" }
}

# ----------------- Start -----------------
Write-Host "Running push.ps1 in: $(Get-Location)" -ForegroundColor Cyan

# 1) Write .gitignore (safe single-quoted here-string => no variable expansion)
$gitignore = @'
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
'@

$gitignorePath = Join-Path -Path (Get-Location) -ChildPath ".gitignore"
Write-Host "`nWriting .gitignore -> $gitignorePath"
$gitignore | Set-Content -Path $gitignorePath -Encoding UTF8

# 2) Ensure git is available
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git not found in PATH. Install Git for Windows and retry."
    exit 1
}

# 3) Initialize repo if missing
if (-not (Test-Path ".git")) {
    Write-Host "`nInitializing new git repository..."
    Run-Git -Args @("init")
} else {
    Write-Host "`nExisting git repository detected."
}

# 4) Ensure remote origin is set correctly
$remoteUrl = "https://github.com/CCC-DATA-factory/Passport-Validation-API.git"
$existingRemote = $null
try {
    $existingRemote = (& git remote get-url origin) -join ""
} catch {
    $existingRemote = $null
}

if ($existingRemote) {
    if ($existingRemote -ne $remoteUrl) {
        Write-Host "Remote 'origin' exists and points to: $existingRemote"
        Write-Host "Setting remote 'origin' to: $remoteUrl"
        Run-Git -Args @("remote","set-url","origin",$remoteUrl)
    } else {
        Write-Host "Remote 'origin' already correctly set."
    }
} else {
    Write-Host "Adding remote origin: $remoteUrl"
    Run-Git -Args @("remote","add","origin",$remoteUrl)
}

# 5) Configure user details if not set
$curName = (& git config user.name) -join ""
$curEmail = (& git config user.email) -join ""
if (-not $curName) {
    Run-Git -Args @("config","user.name","DevButterflies")
} else {
    Write-Host "git user.name = $curName"
}
if (-not $curEmail) {
    Run-Git -Args @("config","user.email","rabii.nasri1@gmail.com")
} else {
    Write-Host "git user.email = $curEmail"
}

# 6) Stage and commit changes if any
Write-Host "`nStaging all changes..."
Run-Git -Args @("add","-A")

$status = (& git status --porcelain) -join "`n"
if ($status) {
    Write-Host "Committing changes..."
    Run-Git -Args @("commit","-m","Initial commit: Passport Validation API + stress tests + READMES")
} else {
    Write-Host "No changes to commit (working tree clean)."
}

# 7) Ensure branch is named main
Run-Git -Args @("branch","-M","main")

# 8) Fetch remote
Write-Host "`nFetching origin..."
$fetch = Run-Git -Args @("fetch","origin")

# 9) Check if origin/main exists
$originMainExists = $false
$lsRemote = Run-Git -Args @("ls-remote","--heads","origin","main")
if ($lsRemote.ExitCode -eq 0 -and $lsRemote.Output.Trim() -ne "") {
    $originMainExists = $true
    Write-Host "Remote branch origin/main exists."
} else {
    Write-Host "Remote branch origin/main does not exist (remote may be empty)."
}

# 10) Try pull --rebase
if ($originMainExists) {
    Write-Host "`nAttempting 'git pull --rebase origin main'..."
    $pull = Run-Git -Args @("pull","--rebase","origin","main")
    if ($pull.ExitCode -eq 0) {
        Write-Host "`nRebase succeeded. Now pushing to origin/main..."
        $push = Run-Git -Args @("push","-u","origin","main")
        if ($push.ExitCode -eq 0) {
            Write-Host "`nPush succeeded." -ForegroundColor Green
            exit 0
        } else {
            Write-Host "`nPush failed after rebase. Attempting 'git push --force-with-lease'..." -ForegroundColor Yellow
            $force = Run-Git -Args @("push","--force-with-lease","origin","main")
            if ($force.ExitCode -eq 0) {
                Write-Host "Force-with-lease push succeeded." -ForegroundColor Green
                exit 0
            } else {
                Write-Error "Push failed even after force-with-lease. Manual intervention required."
                exit 2
            }
        }
    } else {
        Write-Warning "`nRebase failed. Aborting rebase (if present) and attempting merge fallback..."
        # abort rebase if in progress (ignore error)
        try { Run-Git -Args @("rebase","--abort") } catch { }

        # Attempt merge preferring remote changes on conflict
        Write-Host "Merging origin/main into local main while preferring remote changes on conflict..."
        $merge = Run-Git -Args @("merge","origin/main","-m","Auto-merge origin/main (prefer theirs)","-X","theirs")
        if ($merge.ExitCode -eq 0) {
            Write-Host "Merge succeeded. Pushing merged branch..."
            $push2 = Run-Git -Args @("push","-u","origin","main")
            if ($push2.ExitCode -eq 0) {
                Write-Host "Push succeeded after merge." -ForegroundColor Green
                exit 0
            } else {
                Write-Host "Push failed after merge. Attempting force-with-lease..." -ForegroundColor Yellow
                $force2 = Run-Git -Args @("push","--force-with-lease","origin","main")
                if ($force2.ExitCode -eq 0) {
                    Write-Host "Force-with-lease push succeeded." -ForegroundColor Green
                    exit 0
                } else {
                    Write-Error "Push still failed after merge. Manual resolution required."
                    exit 3
                }
            }
        } else {
            Write-Warning "Automatic merge failed. Attempting fallback: create a fallback branch and force-push it to origin/main."
            $timestamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
            $fallbackBranch = "fallback-local-$timestamp"
            Run-Git -Args @("checkout","-b",$fallbackBranch)
            Write-Host "Created fallback branch: $fallbackBranch"

            Write-Host "Attempting 'git push --force-with-lease origin HEAD:main' to update remote main with local content (use with caution)..."
            $force3 = Run-Git -Args @("push","--force-with-lease","origin","HEAD:main")
            if ($force3.ExitCode -eq 0) {
                Write-Host "Force push succeeded. Remote main was replaced by local content." -ForegroundColor Yellow
                Write-Host "If this was not intended, inspect local backups or remote history to restore." -ForegroundColor Yellow
                exit 0
            } else {
                Write-Error "Final force push attempt failed. Manual conflict resolution required."
                exit 4
            }
        }
    }
} else {
    # remote main does not exist -> just push
    Write-Host "`nRemote main does not exist. Pushing local main..."
    $push3 = Run-Git -Args @("push","-u","origin","main")
    if ($push3.ExitCode -eq 0) {
        Write-Host "Push succeeded." -ForegroundColor Green
        exit 0
    } else {
        Write-Error "Push failed even though remote main didn't exist. Inspect output above."
        exit 5
    }
}
