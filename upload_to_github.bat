@echo off
echo ===================================================
echo SpaceGuard AI - GitHub Repository Upload Script
echo Target: https://github.com/keerthivasanr39-crypto/SpaceGuard-AI
echo ===================================================

cd /d "%~dp0"

echo [1/6] Initializing Git repository...
git init

echo [2/6] Staging all project files...
git add .

echo [3/6] Creating commit...
git commit -m "Initial commit: SpaceGuard AI Early Warning System"

echo [4/6] Setting default branch to main...
git branch -M main

echo [5/6] Configuring remote origin...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/keerthivasanr39-crypto/SpaceGuard-AI.git

echo [6/6] Pushing to GitHub...
echo (If prompted, authenticate via GitHub Web Browser or Personal Access Token)
git push -u origin main

echo ===================================================
echo Done! Check https://github.com/keerthivasanr39-crypto/SpaceGuard-AI
echo ===================================================
pause
