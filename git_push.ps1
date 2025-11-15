param(
    [Parameter(Mandatory=$false)] [string] $RemoteUrl,
    [Parameter(Mandatory=$false)] [string] $UserName = "",
    [Parameter(Mandatory=$false)] [string] $UserEmail = "",
    [Parameter(Mandatory=$false)] [string] $CommitMessage = "Initial commit"
)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed or not on PATH. Install Git first: https://git-scm.com/download/win"
    exit 1
}

if (-not $RemoteUrl) {
    $RemoteUrl = Read-Host "Enter remote repository URL (e.g. https://github.com/you/repo.git)"
}

Write-Host "Preparing to push project to: $RemoteUrl" -ForegroundColor Cyan

if ($UserName) { git config --global user.name "$UserName" }
if ($UserEmail) { git config --global user.email "$UserEmail" }

if (-not (Test-Path .git)) {
    git init
    git add .
    git commit -m "$CommitMessage"
    git branch -M main
    git remote add origin $RemoteUrl
} else {
    git add .
    git commit -m "$CommitMessage" -q
}

Write-Host "Pushing to origin main..." -ForegroundColor Green

git push -u origin main

Write-Host "Push complete." -ForegroundColor Green
