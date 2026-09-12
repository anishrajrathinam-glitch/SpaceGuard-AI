# SpaceGuard AI - GitHub Upload Script
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "🛰️ SpaceGuard AI - GitHub Repository Upload Script" -ForegroundColor Cyan
Write-Host "Target: https://github.com/keerthivasanr39-crypto/SpaceGuard-AI" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot

Write-Host "`n[1/6] Initializing Git repository..." -ForegroundColor Green
git init

Write-Host "`n[2/6] Staging project files..." -ForegroundColor Green
git add .

Write-Host "`n[3/6] Creating commit..." -ForegroundColor Green
git commit -m "Initial commit: SpaceGuard AI Early Warning System"

Write-Host "`n[4/6] Setting default branch to main..." -ForegroundColor Green
git branch -M main

Write-Host "`n[5/6] Configuring remote origin..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin https://github.com/keerthivasanr39-crypto/SpaceGuard-AI.git

Write-Host "`n[6/6] Pushing to GitHub (origin main)..." -ForegroundColor Green
Write-Host "Note: If prompted, authenticate via GitHub browser dialog or Personal Access Token (PAT)." -ForegroundColor Yellow
git push -u origin main

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "✅ Upload process completed!" -ForegroundColor Green
Write-Host "Repository: https://github.com/keerthivasanr39-crypto/SpaceGuard-AI" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
