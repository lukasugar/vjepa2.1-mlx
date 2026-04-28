#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCES_DIR="${ROOT_DIR}/references"
TARGET_DIR="${REFERENCES_DIR}/vjepa2"
REPO_URL="https://github.com/facebookresearch/vjepa2"
COMMIT_SHA="204698b45b3712590f06245fbfba32d3be539812"

mkdir -p "${REFERENCES_DIR}"

if [[ -e "${TARGET_DIR}" && ! -d "${TARGET_DIR}/.git" ]]; then
  echo "Refusing to overwrite non-git path: ${TARGET_DIR}" >&2
  exit 1
fi

if [[ -d "${TARGET_DIR}/.git" ]]; then
  CURRENT_REMOTE="$(git -C "${TARGET_DIR}" remote get-url origin 2>/dev/null || true)"
  if [[ "${CURRENT_REMOTE}" != "${REPO_URL}" ]]; then
    echo "Existing repo at ${TARGET_DIR} has unexpected origin: ${CURRENT_REMOTE}" >&2
    exit 1
  fi
else
  git init -q "${TARGET_DIR}"
  git -C "${TARGET_DIR}" remote add origin "${REPO_URL}"
fi

git -C "${TARGET_DIR}" fetch --quiet --depth 1 origin "${COMMIT_SHA}"
git -C "${TARGET_DIR}" -c advice.detachedHead=false checkout --quiet --detach "${COMMIT_SHA}"

echo "vjepa2 reference ready at ${TARGET_DIR}"
echo "pinned commit: ${COMMIT_SHA}"
