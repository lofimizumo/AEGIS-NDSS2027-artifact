# AEGIS NDSS2027 Paper ID: 3053 Artifact

This is a scaled-down model experiment supporting the directional claim that AEGIS
suppresses embedding-gradient token-set leakage vs an undefended baseline.


### Requirements

| Item | Need |
|------|------|
| Hardware | About 8 CPU cores, 16 GB RAM, 5 GB free disk. No GPU. |
| OS | Linux or macOS (Docker works on amd64 or arm64) |
| Python (local install) | Python **3.11** with `venv` |

These package versions are fixed by the install scripts:

```text
torch=2.5.1  transformers=4.46.3  datasets=2.21.0  numpy=1.26.4
```

---

## Installation

Use **Docker** or **local Python**. If you do not have Python 3.11, Docker is easier.

### Option A — Docker

**1. Before you start**

- Install Docker (Docker Desktop is fine on macOS or Windows).
- Keep about 5 GB free disk space.

**2. Build the image** (needs internet):

```bash
docker build --tag aegis-artifact:ndss27 .
```

If the build works, you should see:

```text
Evaluator environment ready: torch=2.5.1, transformers=4.46.3, datasets=2.21.0, numpy=1.26.4
```

The image uses CPU-only PyTorch.

**3. Quick test**

```bash
docker create --name aegis-ktt aegis-artifact:ndss27
docker start --attach aegis-ktt
docker cp aegis-ktt:/opt/aegis/artifact-results/kick-the-tires.json ./kick-the-tires.json
docker rm aegis-ktt
```

**4. Scaled run**

```bash
docker create --name aegis-repro aegis-artifact:ndss27 \
  /opt/aegis/scripts/run-scaled-reproduction.sh
docker start --attach aegis-repro
docker cp aegis-repro:/opt/aegis/artifact-results/scaled-reproduction.json \
  ./scaled-reproduction.json
docker rm aegis-repro
```

If Docker says the name is already in use, remove the old container first:

```bash
docker rm -f aegis-ktt aegis-repro
```

---

### Option B — Local Python 3.11

**1. Install Python 3.11**

On macOS with Homebrew:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 --version   # Apple Silicon
# or: /usr/local/bin/python3.11 --version  # Intel Mac
```

On Linux:

```bash
python3.11 --version
```

**2. Run the install script** from this folder (needs internet).

If `python3.11` is already on your PATH:

```bash
scripts/install-artifact.sh
```

If not, set the path:

```bash
AEGIS_PYTHON=/opt/homebrew/bin/python3.11 scripts/install-artifact.sh
# Intel Mac example:
# AEGIS_PYTHON=/usr/local/bin/python3.11 scripts/install-artifact.sh
```


If install works, you should see:

```text
Evaluator environment ready: torch=2.5.1, transformers=4.46.3, datasets=2.21.0, numpy=1.26.4
Installation complete. Run .../scripts/kick-the-tires.sh
```

**3. Turn on the venv**

```bash
source .venv/bin/activate
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)"
```

You should see Python 3.11.x and torch 2.5.1 with `cuda = None`.

**4. Run the artifact**

Menu:

```bash
aegis artifact
# [1] quick-test   [2] scaled-reproduction   [q] quit
```

Or run one profile:

```bash
aegis artifact quick-test              # usually under 2 minutes
aegis artifact scaled-reproduction     # about 5 minutes; shows a progress bar
```

Or use the helper scripts (they also check the result file):

```bash
scripts/kick-the-tires.sh
scripts/run-scaled-reproduction.sh
```

Results go here by default:

```text
artifact-results/kick-the-tires.json
artifact-results/scaled-reproduction.json
```

To pick another output folder:

```bash
scripts/kick-the-tires.sh /absolute/path/to/output
AEGIS_ARTIFACT_RESULTS_DIR=/absolute/path/to/output scripts/run-scaled-reproduction.sh
```

**5. Check a result file again** (optional)

```bash
./.venv/bin/python scripts/verify-artifact.py \
  --profile kick-the-tires artifact-results/kick-the-tires.json
./.venv/bin/python scripts/verify-artifact.py \
  --profile scaled-reproduction artifact-results/scaled-reproduction.json
```

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `Python 3.11 was not found` | Install Python 3.11, or set `AEGIS_PYTHON` to its full path |
| Installer rejects Python | Use **3.11.x** only |
| Old or broken `.venv` | Run `mv .venv .venv.bak`, then install again |
| CUDA torch error | Delete the venv and use the install script (CPU only) |
| Docker `groupadd` / `useradd` fails | Rebuild from this folder |
| Container name already in use | `docker rm -f aegis-ktt` or `docker rm -f aegis-repro` |
| Download fails during setup | Check internet access to Docker Hub / PyPI |

---

### What “pass” means

- **kick-the-tires:** AEGIS runs, the defense channels change the gradients enough, and losses stay valid numbers.
- **scaled-reproduction:** same checks, plus open recall ≥ 0.9 and AEGIS-protected recall ≤ 0.05 on content tokens.

The official path uses **CPU only** and one thread.

## Extra research studies (optional)

```bash
aegis describe configs/studies/adaptive-attack-evaluation.json
aegis run configs/studies/adaptive-attack-evaluation.json --quick
```

These may download models or data.
