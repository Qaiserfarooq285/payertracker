#!/usr/bin/env bash
# PitchVision -- set up (or update) the Hostinger KVM VPS as the always-on front door
# (docs/DEPLOY.md "Always-on gateway"). Idempotent: the first run installs everything, every later
# run pulls the latest code and restarts the gateway.
#
#   DOMAIN=app.example.com GITHUB_TOKEN=... RUNPOD_API_KEY=... PV_ACCESS_PASSWORD=... \
#     [EMAIL=you@example.com] [PV_POD_SSH_PUBLIC_KEY="ssh-ed25519 ..."] [GEMINI_API_KEY=...] \
#     bash setup_vps.sh
#
#   DOMAIN               the hostname whose A record points at this VPS (required)
#   PV_CODE_BUNDLE       path to a `git bundle` of the branch (docker/deploy_vps.sh ships one from
#                        the owner's Mac, 2026-09-26) -- the code then never comes from GitHub here
#   GITHUB_TOKEN         read-only PAT for the private repo -- only needed without PV_CODE_BUNDLE
#                        (pods no longer use it: they fetch their code from this gateway)
#   RUNPOD_API_KEY       pods read/write (required on first run)
#   PV_ACCESS_PASSWORD   the site login (required on first run)
#   EMAIL                Let's Encrypt expiry notices (optional)
#   PV_BRANCH            branch to run, default master
#
# Values given are written to /etc/pitchvision/gateway.env; ones left out keep whatever that file
# already has, so an update run needs only DOMAIN. Runs as root on Ubuntu/Debian.
set -euo pipefail

: "${DOMAIN:?set DOMAIN=app.example.com}"
BRANCH="${PV_BRANCH:-master}"
REPO_URL="https://github.com/Qaiserfarooq285/payertracker.git"
APP=/opt/pitchvision
VENV="$APP/.venv-gateway"
ENV_DIR=/etc/pitchvision
ENV_FILE="$ENV_DIR/gateway.env"
DATA_DIR=/var/lib/pitchvision
UNIT=pitchvision-gateway

log() { echo "[pitchvision-vps] $*"; }

export DEBIAN_FRONTEND=noninteractive
need=()
for pkg in nginx certbot python3-certbot-nginx python3-venv git; do
  dpkg -s "$pkg" >/dev/null 2>&1 || need+=("$pkg")
done
if [ "${#need[@]}" -gt 0 ]; then
  log "installing ${need[*]}"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "${need[@]}" >/dev/null
fi

# --- secrets file: merge what was passed in over what is already there --------------------
mkdir -p "$ENV_DIR" "$DATA_DIR/input"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"
set_env() {  # set_env NAME VALUE -- replaces or appends NAME=VALUE in the env file
  local name="$1" value="$2"
  if grep -q "^${name}=" "$ENV_FILE"; then
    python3 - "$ENV_FILE" "$name" "$value" <<'PY'
import sys, pathlib
path, name, value = sys.argv[1:]
lines = pathlib.Path(path).read_text().splitlines()
lines = [f"{name}={value}" if l.startswith(f"{name}=") else l for l in lines]
pathlib.Path(path).write_text("\n".join(lines) + "\n")
PY
  else
    printf '%s=%s\n' "$name" "$value" >>"$ENV_FILE"
  fi
}
for name in GITHUB_TOKEN RUNPOD_API_KEY PV_ACCESS_PASSWORD PV_POD_SSH_PUBLIC_KEY GEMINI_API_KEY \
            PV_GPU_TYPES PV_DATACENTER PV_POD_NAME PV_VOLUME_NAME PV_IDLE_STOP_MINUTES \
            PV_GPU_IDLE_MINUTES; do
  if [ -n "${!name:-}" ]; then set_env "$name" "${!name}"; fi
done
grep -q '^PV_GATEWAY_DATA=' "$ENV_FILE" || set_env PV_GATEWAY_DATA "$DATA_DIR"
# Pods fetch their code from this gateway's public URL (apps/gateway/main.py `/pod/*`).
set_env PV_PUBLIC_URL "https://$DOMAIN"
for required in RUNPOD_API_KEY PV_ACCESS_PASSWORD; do
  grep -q "^${required}=." "$ENV_FILE" || { log "$required is not set (pass it on the command line)"; exit 1; }
done

# --- code --------------------------------------------------------------------------------------
if [ -n "${PV_CODE_BUNDLE:-}" ] && [ -f "$PV_CODE_BUNDLE" ]; then
  # Shipped from the owner's Mac -- no GitHub credentials needed on this machine.
  if [ ! -d "$APP/.git" ]; then
    log "cloning from the shipped bundle -> $APP"
    git clone --quiet --branch "$BRANCH" "$PV_CODE_BUNDLE" "$APP"
  else
    log "updating $APP from the shipped bundle"
    git -C "$APP" fetch --quiet "$PV_CODE_BUNDLE" "$BRANCH"
    git -C "$APP" reset --hard --quiet FETCH_HEAD
  fi
else
  # Private repo: the token rides on a per-command header, never in .git/config.
  grep -q '^GITHUB_TOKEN=.' "$ENV_FILE" || { log "no PV_CODE_BUNDLE and no GITHUB_TOKEN -- cannot get the code"; exit 1; }
  TOKEN="$(sed -n 's/^GITHUB_TOKEN=//p' "$ENV_FILE")"
  GIT_AUTH=(-c "http.https://github.com/.extraheader=Authorization: Basic $(printf 'x-access-token:%s' "$TOKEN" | base64 -w0)")
  if [ ! -d "$APP/.git" ]; then
    log "cloning $REPO_URL ($BRANCH) -> $APP"
    git "${GIT_AUTH[@]}" clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP"
  else
    log "updating $APP to origin/$BRANCH"
    git "${GIT_AUTH[@]}" -C "$APP" fetch --quiet origin "$BRANCH"
    git -C "$APP" reset --hard --quiet "origin/$BRANCH"
  fi
fi
log "running $(git -C "$APP" rev-parse --short HEAD): $(git -C "$APP" log -1 --pretty=%s)"

# --- python env for the gateway only (no torch, no pipeline deps) ------------------------------
[ -x "$VENV/bin/python" ] || { log "creating $VENV"; python3 -m venv "$VENV"; }
STAMP="$VENV/.deps-$(sha256sum "$APP/apps/gateway/requirements.txt" | cut -c1-16)"
if [ ! -f "$STAMP" ]; then
  log "installing gateway dependencies"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$APP/apps/gateway/requirements.txt"
  rm -f "$VENV"/.deps-*
  touch "$STAMP"
fi

# --- systemd service --------------------------------------------------------------------------
install -m 0644 "$APP/docker/vps/$UNIT.service" "/etc/systemd/system/$UNIT.service"
systemctl daemon-reload
systemctl enable --now "$UNIT" >/dev/null
systemctl restart "$UNIT"

# --- nginx vhost ---------------------------------------------------------------------------
log "writing vhost for $DOMAIN -> 127.0.0.1:8100"
mkdir -p /var/www/pitchvision
install -m 0644 "$APP/docker/vps/gateway-offline.html" /var/www/pitchvision/gateway-offline.html
sed -e "s|__DOMAIN__|$DOMAIN|g" "$APP/docker/vps/pitchvision.nginx.conf" >/etc/nginx/sites-available/pitchvision
ln -sf /etc/nginx/sites-available/pitchvision /etc/nginx/sites-enabled/pitchvision
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx >/dev/null
systemctl reload nginx

# Certificate: only once the A record resolves here, otherwise certbot's HTTP challenge fails.
if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
  log "requesting Let's Encrypt certificate for $DOMAIN"
  if [ -n "${EMAIL:-}" ]; then
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect
  else
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email --redirect
  fi
else
  # The vhost was just rewritten without the TLS block certbot adds -- re-apply it (a no-op on
  # the certificate itself while it is still valid).
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --reinstall --redirect >/dev/null
  systemctl reload nginx
fi

if [ -f /etc/ufw/ufw.conf ] && ufw status | grep -q "Status: active"; then
  ufw allow 'Nginx Full' >/dev/null || true
fi

# --- did it come up? ----------------------------------------------------------------------
for _ in $(seq 1 20); do
  if curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/api/health | grep -q '^200$'; then
    log "gateway is up: $(curl -s -m 5 http://127.0.0.1:8100/api/health)"
    log "done: https://$DOMAIN"
    exit 0
  fi
  sleep 2
done
log "gateway did not answer on :8100 -- journalctl -u $UNIT -n 50"
journalctl -u "$UNIT" -n 30 --no-pager || true
exit 1
