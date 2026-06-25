#!/bin/bash
# ================================================
# Push all Elena's projects to GitHub
# Run this from the github_repos/ folder
# Replace YOUR_USERNAME with your GitHub username
# ================================================

GITHUB_USERNAME="elenakinael-del"

declare -A REPOS=(
  ["gold-psychophysics-cot"]="gold-psychophysics-cot"
  ["neuro-quant-trading"]="neuro-quant-trading"
  ["xauusd-backtest"]="xauusd-backtest"
  ["kuramoto-herding-model"]="kuramoto-herding-model"
  ["gold-psychophysical-volatility"]="gold-psychophysical-volatility"
  ["neural-market-visualizer"]="neural-market-visualizer"
)

for folder in "${!REPOS[@]}"; do
  repo="${REPOS[$folder]}"
  echo ""
  echo "=========================================="
  echo "Pushing: $folder → github.com/$GITHUB_USERNAME/$repo"
  echo "=========================================="
  cd "$folder"
  git init
  git add .
  git commit -m "Initial commit: $repo"
  git branch -M main
  git remote add origin "https://github.com/$GITHUB_USERNAME/$repo.git"
  git push -u origin main
  cd ..
done

echo ""
echo "ALL DONE! Check your GitHub profile."
