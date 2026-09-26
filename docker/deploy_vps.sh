#!/usr/bin/env bash
# PitchVision -- ship this checkout's master to the VPS gateway (docs/DEPLOY.md "Always-on gateway").
# The code travels as a `git bundle` over SSH (2026-09-26), so neither the VPS nor the pods need a
# GitHub token any more; the pod side needs nothing -- it fetches its code from the gateway on its
# next Start GPU. Run from the owner's Mac after committing (and `git push`, to keep GitHub level):
#
#   bash docker/deploy_vps.sh
#
# Reads (all already on the owner's machine after the first deployment):
#   ~/.ssh/pitchvision_github_token.txt   GITHUB_TOKEN (optional now; kept for the GitHub fallback)
#   ~/.ssh/pitchvision_site_password.txt  PV_ACCESS_PASSWORD
#   ~/.ssh/pitchvision_runpod_key.txt     RUNPOD_API_KEY (or the env var)
#   ~/.ssh/pitchvision_vps(.pub)          SSH key for the VPS; the .pub is installed on the pod too
set -euo pipefail

DOMAIN="${DOMAIN:-app.thereachvision.tech}"
VPS="${VPS:-root@187.6.165.147}"
VPS_KEY="${VPS_KEY:-$HOME/.ssh/pitchvision_vps}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "$HOME/.ssh/pitchvision_runpod_key.txt")}"
GITHUB_TOKEN="${GITHUB_TOKEN:-$(cat "$HOME/.ssh/pitchvision_github_token.txt" 2>/dev/null || true)}"
BRANCH="${PV_BRANCH:-master}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASSWORD="$(cat "$HOME/.ssh/pitchvision_site_password.txt")"
SSH_PUB="$(cat "$VPS_KEY.pub")"

log() { echo "[deploy-vps] $*"; }
ssh_vps() { ssh -i "$VPS_KEY" -o StrictHostKeyChecking=accept-new "$VPS" "$@"; }

# The setup script lives in the repo; the VPS keeps its own checkout, but the FIRST run needs the
# script before any checkout exists -- so always ship the current copy over first.
log "copying setup script to the VPS"
ssh_vps "mkdir -p /root/pitchvision-vps"
scp -q -i "$VPS_KEY" "$(dirname "${BASH_SOURCE[0]}")/vps/setup_vps.sh" "$VPS:/root/pitchvision-vps/setup_vps.sh"

BUNDLE_DIR="$(mktemp -d)"
BUNDLE="$BUNDLE_DIR/code.bundle"
trap 'rm -rf "$BUNDLE_DIR"' EXIT
log "bundling $BRANCH ($(git -C "$REPO_DIR" rev-parse --short "$BRANCH"))"
git -C "$REPO_DIR" bundle create --quiet "$BUNDLE" "$BRANCH"
scp -q -i "$VPS_KEY" "$BUNDLE" "$VPS:/root/pitchvision-vps/code.bundle"

log "running setup on the VPS ($DOMAIN)"
ssh_vps "DOMAIN=$(printf %q "$DOMAIN") PV_BRANCH=$(printf %q "$BRANCH") PV_CODE_BUNDLE=/root/pitchvision-vps/code.bundle \
  ${GITHUB_TOKEN:+GITHUB_TOKEN=$(printf %q "$GITHUB_TOKEN")} RUNPOD_API_KEY=$(printf %q "$RUNPOD_API_KEY") \
  PV_ACCESS_PASSWORD=$(printf %q "$PASSWORD") PV_POD_SSH_PUBLIC_KEY=$(printf %q "$SSH_PUB") \
  ${GEMINI_API_KEY:+GEMINI_API_KEY=$(printf %q "$GEMINI_API_KEY")} \
  bash /root/pitchvision-vps/setup_vps.sh"

log "health: $(curl -s -m 15 "https://$DOMAIN/api/health")"
