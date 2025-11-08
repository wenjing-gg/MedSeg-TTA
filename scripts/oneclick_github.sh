#!/usr/bin/env bash
set -euo pipefail

# One-click create and push MedSegTTABoard to GitHub.
# Supports three paths: gh CLI, HTTPS + PAT, SSH.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_NAME=${REPO_NAME:-MedSegTTABoard}
VISIBILITY=${VISIBILITY:-public}   # public|private|internal (user repo不支持internal)
DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}

# Owner: your GitHub username or org name
GITHUB_OWNER=${GITHUB_OWNER:-}

# If using PAT: export GITHUB_TOKEN with repo scope
GITHUB_TOKEN=${GITHUB_TOKEN:-}
# Or provide a file path via GITHUB_TOKEN_FILE to avoid exposing in process list/logs
GITHUB_TOKEN_FILE=${GITHUB_TOKEN_FILE:-}

# Use SSH remote? set USE_SSH=1
USE_SSH=${USE_SSH:-0}

need() { command -v "$1" >/dev/null 2>&1; }

echo "[1/5] Init local git repo and first commit"
cd "$REPO_DIR"
if [ ! -d .git ]; then
  git init -q
  git checkout -q -b "$DEFAULT_BRANCH"
fi
# Ensure local git identity (fallback if global unset)
if ! git config user.name >/dev/null; then
  git config user.name "MedSegTTABoard"
fi
if ! git config user.email >/dev/null; then
  git config user.email "noreply@example.com"
fi
git add -A
if ! git rev-parse HEAD >/dev/null 2>&1; then
  git commit -m "init: MedSegTTABoard initial commit" >/dev/null
else
  if ! git diff --cached --quiet; then
    git commit -m "chore: sync initial content" >/dev/null
  fi
fi

echo "[2/5] Determine remote create method (gh | PAT | manual)"
REMOTE_URL=""

if need gh; then
  if gh auth status >/dev/null 2>&1; then
    if [ -z "${GITHUB_OWNER}" ]; then
      # Try to infer owner from gh auth status
      GITHUB_OWNER=$(gh api user --jq .login)
    fi
    echo "Using gh to create repo: ${GITHUB_OWNER}/${REPO_NAME} (${VISIBILITY})"
    gh repo create "${GITHUB_OWNER}/${REPO_NAME}" --${VISIBILITY} --source . --remote origin --push
    echo "Done. URL: https://github.com/${GITHUB_OWNER}/${REPO_NAME}"
    exit 0
  fi
fi

if [ -n "${GITHUB_TOKEN_FILE}" ] && [ -z "${GITHUB_TOKEN}" ]; then
  if [ -f "${GITHUB_TOKEN_FILE}" ]; then
    GITHUB_TOKEN=$(cat "${GITHUB_TOKEN_FILE}")
  else
    echo "GITHUB_TOKEN_FILE 不存在: ${GITHUB_TOKEN_FILE}" >&2
    exit 1
  fi
fi

if [ -n "${GITHUB_TOKEN}" ] && [ -n "${GITHUB_OWNER}" ]; then
  echo "Using GitHub API with PAT to create repo: ${GITHUB_OWNER}/${REPO_NAME} (${VISIBILITY})"
  API_URL="https://api.github.com"
  # Detect if owner is a user or org
  OWNER_TYPE=$(curl -sS -H "Authorization: token ${GITHUB_TOKEN}" "${API_URL}/users/${GITHUB_OWNER}" | jq -r .type)
  if [ "${OWNER_TYPE}" = "Organization" ]; then
    CREATE_ENDPOINT="${API_URL}/orgs/${GITHUB_OWNER}/repos"
    DATA=$(jq -n --arg name "$REPO_NAME" --arg vis "$VISIBILITY" '{name:$name, private: ($vis != "public") }')
  else
    CREATE_ENDPOINT="${API_URL}/user/repos"
    DATA=$(jq -n --arg name "$REPO_NAME" --arg vis "$VISIBILITY" '{name:$name, private: ($vis != "public") }')
  fi
  curl -sS -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" \
       -d "${DATA}" "${CREATE_ENDPOINT}" >/dev/null

  if [ "${USE_SSH}" = "1" ]; then
    REMOTE_URL="git@github.com:${GITHUB_OWNER}/${REPO_NAME}.git"
  else
    # For one-shot push via HTTPS, embed token in URL (avoid storing persistently)
    REMOTE_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_OWNER}/${REPO_NAME}.git"
  fi
else
  echo "gh 未登录，且未提供 GITHUB_TOKEN/GITHUB_OWNER。进入手动远端配置："
  echo "- 请设置环境变量 GITHUB_OWNER=你的 GitHub 用户名或组织名"
  echo "- 若使用 PAT，请导出 GITHUB_TOKEN=你的 token (repo 权限)"
  echo "- 若使用 SSH，请先完成 ssh-key 配置并将 USE_SSH=1"
  echo "示例： GITHUB_OWNER=yourname GITHUB_TOKEN=xxx VISIBILITY=public USE_SSH=1 scripts/oneclick_github.sh"
  exit 1
fi

echo "[3/5] Set remote origin: ${REMOTE_URL}"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

echo "[4/5] Push to remote"
git push -u origin "$DEFAULT_BRANCH"

echo "[5/5] Done"
if [ "${USE_SSH}" = "1" ]; then
  echo "Repo URL: https://github.com/${GITHUB_OWNER}/${REPO_NAME}"
else
  # Clean remote to a safe HTTPS URL without embedding the token
  git remote set-url origin "https://github.com/${GITHUB_OWNER}/${REPO_NAME}.git"
  echo "Repo URL: https://github.com/${GITHUB_OWNER}/${REPO_NAME}"
  echo "已将 origin 重置为不含 token 的 HTTPS URL。若需要，后续可改为 SSH："
  echo "  git remote set-url origin git@github.com:${GITHUB_OWNER}/${REPO_NAME}.git"
fi
