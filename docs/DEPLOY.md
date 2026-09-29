# Hosting The Reach Vision (PitchVision) on RunPod with your own domain

This is the step-by-step for the owner's setup: **RunPod credit** for the GPU and a **Hostinger
domain** for the address. Nothing here needs a Docker image build or a server you have to
maintain — a RunPod pod pulls the code straight from GitHub on every boot.

```
  browser ──https──▶ Cloudflare (your domain, free)
                        │  Cloudflare Tunnel (outbound from the pod; no open ports, no pod IP)
                        ▼
                   RunPod GPU pod  ── runs docker/runpod_bootstrap.sh
                        │             └─ FastAPI + web UI on :8000  (start_app.py, no --reload)
                        ▼
                   /workspace  (RunPod NETWORK VOLUME — survives stop/restart)
                     └─ pitchvision/   checkout · .venv · models/ · input/ · work/ · output/
```

Why this shape:
- **The owner's live setup is §4-VPS** — an always-on gateway on the Hostinger VPS in front of the
  pod, so the site is up even when the pod is stopped or RunPod has no GPU. The Cloudflare Tunnel
  path (§4-CF) is the no-VPS alternative and keeps the pod-only shape drawn above.
- **Pod + network volume, not serverless.** A run takes minutes and the UI polls a long-lived
  server; serverless workers are for short stateless calls. The volume is what lets you **stop
  the pod to stop paying** without losing uploads, outputs or the 1.9 GB of model weights.
- **Cloudflare Tunnel, not DNS → pod IP.** A pod's IP and ports change on every restart and it
  has no port 443; the tunnel gives a stable `https://app.yourdomain.com` for free.
- **Password gate in the app** (`PV_ACCESS_PASSWORD`). The RunPod proxy URL is public; without
  the gate anyone who finds it can upload videos and burn your credit.
- **Chunked uploads.** Cloudflare rejects any single request over 100 MB; the UI now uploads in
  24 MB slices, so 4K clips work through the domain.

---

## 0. Before you start (5 min)

| You need | Where |
|---|---|
| The repo on GitHub (**private**) + a read-only token for it | `github.com/Qaiserfarooq285/payertracker` (`master`). Token: GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained** → Repository access: *only* payertracker → Permissions: **Contents: Read-only** → Generate. Goes into the pod's `GITHUB_TOKEN` env var; nothing else ever sees it. |
| RunPod account with credit | runpod.io |
| Cloudflare account (free) | dash.cloudflare.com — sign up |
| Access to the Hostinger domain's DNS settings | hpanel.hostinger.com |
| A long access password you'll set on the pod | make one up now; you'll type it into the UI |

Optional: `GEMINI_API_KEY` (only the celebration / key-moment stage uses it).

---

## 1. RunPod — network volume (2 min)

1. RunPod → **Storage** → **New Network Volume**.
2. Pick a **region that lists the GPU you want** (volumes are region-locked; check *Pods →
   Deploy* first to see which region has, e.g., RTX 4090 availability).
3. Size **60 GB** (models 2 GB + Python env 7 GB + your videos and outputs). You can grow it
   later; you cannot shrink it. Cost is per GB-month, roughly the price of a coffee.
4. Name it `pitchvision-data`.

## 2. RunPod — pod template (5 min)

RunPod → **Templates** → **New Template**:

| Field | Value |
|---|---|
| Template type | Pod |
| Container image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (any `runpod/pytorch` CUDA ≥ 12.4 image works — the bootstrap builds its own Python env) |
| Container disk | 20 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000` |
| Expose TCP ports | `22` (optional, for SSH debugging) |
| **Container start command** | `bash -c "curl -fsSL -H \"Authorization: token $GITHUB_TOKEN\" https://raw.githubusercontent.com/Qaiserfarooq285/payertracker/master/docker/runpod_bootstrap.sh \| bash"` — copy it from the top of [`docker/runpod_bootstrap.sh`](../docker/runpod_bootstrap.sh) without the `\|` escaping. `$GITHUB_TOKEN` is expanded by the pod at boot, not by you. |
| Environment variables | see [`docker/runpod.env.example`](../docker/runpod.env.example): **`GITHUB_TOKEN`** (required — private repo), **`PV_ACCESS_PASSWORD`** (optional — defaults to `admin1122` if left unset, change it), `RUNPOD_API_KEY` (optional, enables idle auto-stop — §"Idle auto-stop" below), `CLOUDFLARE_TUNNEL_TOKEN` (after step 4), `GEMINI_API_KEY` (optional) |

Save the template.

## 3. RunPod — deploy the pod (first boot ≈ 10 min)

1. **Pods → Deploy**. GPU: the pipeline was built for a 12 GB card, so anything from
   **RTX A4000 (16 GB)** up works; **RTX 3090 / 4090 (24 GB)** are the sweet spot for speed
   per dollar. Prefer *Secure Cloud* for reliability; *Community Cloud* is cheaper.
2. **Select your network volume** (`pitchvision-data`) — this is the step people miss; without
   it everything is wiped on restart.
3. Choose the template from step 2 → **Deploy**.
4. Open the pod's **Logs**. You'll see `[pitchvision] cloning…`, `installing dependencies
   (first boot: several minutes)`, `downloading …`, then `starting server on port 8000`.
5. **Connect → HTTP Service [Port 8000]** opens `https://<POD_ID>-8000.proxy.runpod.net`.
   Enter your access password. Upload a clip, run it — this is the full app on the GPU.

Every later boot skips the install and starts in ~30 s.

### Login

The web UI is gated by a single shared password — the whole point is "after deployment it
should be used by us only", not a per-user account system. **The gate is ON out of the box**:
leave `PV_ACCESS_PASSWORD` unset and the pod accepts the built-in default, **`admin1122`**. Set
`PV_ACCESS_PASSWORD` on the template to your own password to change it (Restart to apply).
`PV_ACCESS_PASSWORD=off` disables the gate entirely — only ever do that for local editing on
your own machine, never on a hosted pod.

## 4. Your Hostinger domain → the pod

Two ways to put the pod behind your own `https://app.yourdomain.com`. **4-VPS** is what the
owner runs (a Hostinger KVM VPS is already paid for, so no Cloudflare account or nameserver move
is needed); **4-CF** is the Cloudflare Tunnel alternative if you have no VPS.

### 4-VPS. Always-on gateway on the Hostinger KVM VPS (10 min) — what the owner runs

```
browser ──https──▶ KVM VPS: nginx (TLS) ──▶ gateway :8100 (apps/gateway, systemd)
                                              │  frontend · login · uploads (VPS disk) · job queue
                                              │  finds the pod BY NAME via the RunPod API
                                              ▼
                                   <pod id>-8000.proxy.runpod.net  (whatever pod exists right now)
```

**Why a gateway and not plain nginx** (owner, 2026-09-17): the pod stops itself when idle, and a
*stopped* pod does not reserve its GPU — so Start often fails and, with nginx alone, visitors saw a
"pod is asleep" page and nothing else. The gateway keeps the **site up with no pod at all**:

- The frontend, login and **uploads always work**; uploads land on the VPS disk (`/var/lib/pitchvision/input`).
- **The GPU is switched on and off by the user, nowhere else** (since 2026-09-26 — the old
  "wake on any request, recreate whatever fails, wait forever" policy burned the whole credit):
  - **Start GPU** (the panel under the GPU tiers) creates/starts the pod on the chosen tier. If RunPod
    has no card it retries every minute for at most **15 min**, then gives up (a card freeing up hours
    later must not start billing while nobody is there). An **empty balance** is shown as such, at once.
    A pod that is not up within **20 min** is stopped + terminated and reported — never recreated in a loop.
  - **Stop GPU** cancels queued/running jobs, then stops **and** terminates the pod. Nothing bills.
  - **Idle auto-stop**: a billing pod with no job and no use for `PV_GPU_IDLE_MINUTES` (default 10) is
    stopped by the gateway — this works even when the pod's own app never booted.
  - **Run** is refused while the GPU is off; a job queued while it is starting waits for it, for free,
    and can be cancelled. Page loads, results views and queued jobs never start the GPU.
  - The panel shows the card, its $/h, the auto-stop countdown and the RunPod balance.
- Once the pod answers `/api/health` the gateway pushes the video (skipped when the volume already
  has it), starts the pipeline and mirrors its progress/results/media back to the browser. The job
  log carries the pipeline's real progress lines, so a long video no longer looks frozen.
- **The gateway pushes the code to each new pod**, through the same RunPod proxy it uses for
  everything else. A pod's start command first runs `docker/pod_receiver.py` (it travels in the pod's
  env, `PV_POD_RECEIVER`), which waits on port 8000; the gateway sends it a tarball of the VPS checkout,
  checked against a key derived from the site password, and the pod runs the bundle's
  `docker/runpod_bootstrap.sh`. Neither GitHub nor a token that can expire is in the path (an expired
  one killed every new pod on 2026-09-24). **A pod never calls the VPS**: a pod in EUR-IS-1 cannot open
  a connection to it at all (measured 2026-09-30 — the first real run of the old "pod downloads from
  the gateway" design hung in `curl` for 13 min per try).
- **A pod that stops itself while booting stays stopped.** It stops itself when no code arrives in
  15 min or the bootstrap/app fails; the gateway then reports the failure and terminates it. It never
  starts it again, because on 2026-09-30 the stop-self → restart → stop-self loop billed an A100 with
  nothing running (every restart also reset the 20-min boot cap).
- The queue is persisted (`/var/lib/pitchvision/jobs.json`): a refreshed page — or a rebooted VPS —
  picks up where it was. Jobs run one at a time.
- Because the pod is found **by name**, nothing ever needs re-pointing when its id changes.

1. **DNS** — Hostinger hPanel → Domains → your domain → DNS records → add
   `A  app  <VPS IPv4>  TTL 300`. (`app` = subdomain; use `@` for the bare domain.) Wait until
   `dig +short app.yourdomain.com` returns the VPS IP (usually minutes).
2. **From your Mac**, with the secrets in place (`~/.ssh/pitchvision_runpod_key.txt`,
   `~/.ssh/pitchvision_site_password.txt`, the VPS SSH key `~/.ssh/pitchvision_vps`; a GitHub token
   is no longer needed), run:

   ```bash
   bash docker/deploy_vps.sh
   ```

   It ships your local `master` as a `git bundle` over SSH and runs `docker/vps/setup_vps.sh`:
   installs nginx + certbot + a tiny Python venv, checks the code out to `/opt/pitchvision`, writes `/etc/pitchvision/gateway.env`
   (template: `docker/vps/gateway.env.example`), installs the `pitchvision-gateway` systemd unit,
   writes the vhost from `docker/vps/pitchvision.nginx.conf`, gets the certificate (HTTP challenge —
   so do step 1 first) and turns on the https redirect. **Re-run it after every commit** to
   update the gateway; the pod picks up the same code on its next Start GPU.
3. Open **https://app.yourdomain.com** → the login overlay → your `PV_ACCESS_PASSWORD`.
   `https://app.yourdomain.com/api/health` shows `pod.phase` (`offline` / `starting` / `booting` /
   `waiting_for_gpu` / `online`) and, once online, the GPU the pod sees.

Provisioning the RunPod side by hand instead (§1–3): `RUNPOD_API_KEY=... GITHUB_TOKEN=...
python docker/runpod_provision.py --password '<site password>'` creates the volume + pod and
prints the pod id — but with the gateway in front you never need to: it creates the pod itself the
first time someone presses Run. Network volumes only attach to **Secure Cloud** pods; the pod
must live in the volume's datacenter (`PV_DATACENTER`, default `EUR-IS-1`).

### 4-CF. Cloudflare Tunnel (no VPS; 15 min + DNS wait)


#### 4a. Put the domain on Cloudflare
1. Cloudflare dashboard → **Add a site** → type your domain → **Free** plan → Continue.
   Cloudflare scans and imports existing DNS records; Continue.
2. It shows **two nameservers** (like `ada.ns.cloudflare.com` / `rob.ns.cloudflare.com`).
3. Hostinger hPanel → **Domains** → your domain → **DNS / Nameservers** → **Change
   nameservers** → *Custom* → paste the two Cloudflare nameservers → Save.
4. Wait for Cloudflare's "site is active" email (usually under an hour, up to 24 h).
   Email/website you already host at Hostinger keeps working — the imported records carry over.

#### 4b. Create the tunnel
1. **one.dash.cloudflare.com** (Zero Trust) → **Networks → Tunnels → Create a tunnel** →
   *Cloudflared* → name `pitchvision` → Save.
2. On the install page, ignore the OS buttons; in the command shown, copy the long string after
   `--token`. That is your `CLOUDFLARE_TUNNEL_TOKEN`.
3. **Public hostname** tab → Add: subdomain `app`, domain yours, type **HTTP**, URL
   `localhost:8000` → Save.
4. RunPod → your template (or the running pod's *Edit*) → add env var
   `CLOUDFLARE_TUNNEL_TOKEN=<the token>` → **Restart** the pod.
5. Open **https://app.yourdomain.com**. Cloudflare provides the HTTPS certificate automatically.

#### 4c. Optional extra lock (recommended if others will use it)
Zero Trust → **Access → Applications → Add** → Self-hosted → domain `app.yourdomain.com` →
policy *Allow* → include *Emails* = the addresses you permit. Visitors then get a one-time email
code before they even reach the app's own password. Free up to 50 users. (The RunPod proxy URL
bypasses Cloudflare, which is why the in-app password still matters.)

---

## Day-to-day

| Task | How |
|---|---|
| **Stop paying** when not in use | Press **Stop GPU**. If you forget, the gateway's idle auto-stop (10 min) and the pod's own (30 min) stop it. It only comes back when someone presses **Start GPU**. |
| Update the app | commit on `master` → `git push` → `bash docker/deploy_vps.sh` (gateway) → the pod gets the same code on its next Start GPU. |
| See what the pod is doing | `https://app.yourdomain.com/api/pod` (after login): phase, pod id, GPU, cost/h, the active queue. |
| Change the password | edit `PV_ACCESS_PASSWORD` on the pod → Restart. Sessions are invalidated on every restart anyway. |
| Free disk | SSH / web terminal: `rm -rf /workspace/pitchvision/work/*` (cached intermediates; re-created on demand). Delete old `output/<slug>/` folders you no longer need. |
| Watch logs | Pod → Logs; tunnel log at `/workspace/cloudflared.log`; RunPod's own services at `/workspace/runpod-start.log`. |
| Health | `https://app.yourdomain.com/api/health` (public, no login) shows the GPU the server sees. |

### Idle auto-stop (recommended — saves credit)

Forgetting to press Stop is the single most expensive mistake with a metered pod. Set it up once
and the pod stops itself:

1. RunPod → **Settings → API Keys → Create API Key** → type *Restricted* → give it the **pods**
   permission, read/write → Create → copy the key.
2. Paste it into the template's `RUNPOD_API_KEY` env var → Restart the pod. `RUNPOD_POD_ID` needs
   no setup — RunPod injects it into every pod automatically.
3. That's it. The pod now stops itself after `PV_IDLE_STOP_MINUTES` (default 30) with **no HTTP
   requests and no pipeline job running** — checked every minute in the background
   (`apps/api/idle_stop.py`). Nothing is lost: the network volume (checkout, `input/`, `work/`,
   `output/`) persists across a stop exactly like a manual Stop. The gateway (§4-VPS) starts it
   again on the next **Run**; without the gateway, Start it from the RunPod console.
4. `https://app.yourdomain.com/api/health` → `idle_stop.enabled` confirms it's armed (`false`
   until `RUNPOD_API_KEY` is filled in — off by default, matching the blank env template).

## Pod won't start after a stop ("not enough free GPUs on the host machine")

A **stopped pod does not reserve its GPU**. After an idle auto-stop or a manual Stop, someone else
may take the card, and Start fails with that message (the console then offers "migrate your pod").
Nothing is lost — the network volume is separate from the pod.

**With the gateway (§4-VPS) this is handled for you:** Stop GPU terminates the pod, so the next
Start GPU simply creates a fresh one on the chosen tier; if RunPod has none of those cards it keeps
asking every minute for up to 15 min, then gives up and says so. The state machine is
`apps/gateway/pod_manager.py`; `/api/pod` shows what it is doing and `journalctl -u
pitchvision-gateway -f` on the VPS shows why.

Without the gateway: terminate the pod in the console and run `docker/runpod_provision.py` again
(the volume, and everything on it, carries over).

## Troubleshooting

- **Login overlay never accepts the password** → the pod's `PV_ACCESS_PASSWORD` has trailing
  spaces or you edited it without restarting. Check the pod log's startup banner: `Access gate: ON`
  (or `ON (default password...)` if you left it unset — the gate is on either way).
- **`/api/health` says `gpu_available: false`** → the pod was deployed without a GPU or the
  driver is older than the CUDA 12.4 wheels; redeploy on a different host.
- **Upload dies at ~100 MB** → you're hitting Cloudflare's per-request cap with the old
  single-request path. The current UI uses `/api/upload/chunk`; hard-refresh the page
  (Ctrl+Shift+R) to drop the cached old `app.js`.
- **`Job failed` right after a restart** → any run in flight dies when the pod restarts. Don't
  restart mid-run; results already written to `output/` are kept.
- **Boot dies immediately, log shows `curl: (22) ... 404`** → the repo is private and the pod has no
  (or an expired/wrong) `GITHUB_TOKEN`. Fine-grained tokens expire — generate a new one, update the
  pod env, Restart. Rotating it needs no code change: the token is never written to the volume.
- **First boot stuck on `installing dependencies`** → normal for 5–10 min. If it exceeds 20 min,
  check the log for a pip error and restart; the install resumes from uv's cache.
- **Tunnel shows "inactive"** → the token env var is missing/wrong on the pod, or the pod is
  stopped. `cat /workspace/cloudflared.log`.
- **Choosing the GPU.** The web app's *GPU for this run* picker offers three tiers
  (`apps/gateway/runpod_pods.py: GPU_TIERS` — Budget RTX 4000 Ada, Standard RTX 4090, Pro A100 80 GB)
  with RunPod's **live** Secure Cloud per-hour price and per-datacenter stock (`GET /api/gpus`, cached
  5 min, `PV_GPU_PRICE_TTL_SECONDS`). A run on a tier whose card differs from the pod's current one
  **replaces the pod** (terminate + create on the new card; the volume is untouched) — expect a
  few minutes' boot. Blackwell cards (RTX 5090 / RTX PRO) are deliberately absent: the pod image is
  CUDA 12.4.
- **Domain shows "The Reach Vision is restarting"** → the gateway service is down. On the VPS:
  `systemctl status pitchvision-gateway`, `journalctl -u pitchvision-gateway -n 50`. It restarts
  itself; if it can't start, the log names the missing env var in `/etc/pitchvision/gateway.env`.
- **Badge stuck on "No GPU available right now"** → RunPod genuinely has none of `PV_GPU_TYPES`
  in `PV_DATACENTER`. Widen the list in `/etc/pitchvision/gateway.env` (the volume pins the
  datacenter) and `systemctl restart pitchvision-gateway`; queued jobs survive the restart.
- **Badge says "GPU pod error"** → RunPod rejected the API key (`RUNPOD_API_KEY` in
  `gateway.env` is wrong or lacks pods read/write). Fix it and restart the gateway.
- **"GPU pod stopped answering mid-job"** → the pod died under a run (host taken by RunPod, OOM).
  Press Run again; cached `work/` on the volume makes the re-run resume.
- **`certbot` fails on the VPS** → the `A` record hasn't propagated to the VPS yet
  (`dig +short app.yourdomain.com` must print the VPS IP), or port 80 is firewalled.
- **`409 another run is already in flight`** → the same video is being processed already; wait
  for it or process a different upload (one GPU, one job at a time by design).

## Costs, honestly (check current RunPod prices — they move)

- GPU pod: roughly **$0.20–0.70 / hour while running**, by card. Stop it when idle, or set up
  **idle auto-stop** above so a forgotten tab doesn't run the meter overnight.
- Network volume: **~$0.07 / GB-month** → 60 GB ≈ $4 / month.
- Cloudflare (domain proxy, tunnel, Access ≤ 50 users): **$0**.
- Hostinger: whatever the domain renewal already costs; no hosting plan is needed for this.

## Running it on your own PC instead (unchanged)

`make serve` (or `.venv/bin/python start_app.py`) — no password, no tunnel, `http://localhost:8000`.
Set `PV_DEV=1` only while editing code (auto-reload); it kills in-flight jobs otherwise.
