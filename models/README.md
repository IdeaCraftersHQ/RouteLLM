# Models — ONNX Export & Triton Serving

## Quick start (dev environment)

```bash
# Build and run Triton with exported models
docker compose -f models/docker-compose.yml up --build

# Verify readiness
curl http://localhost:8000/v2/health/ready
```

Triton exposes:
- HTTP: `localhost:8000`
- gRPC: `localhost:8001`
- Metrics: `localhost:8002`

## Export models manually

Requires Python 3.11+ and deps from `models/export/requirements.txt`.

```bash
pip install -r models/export/requirements.txt

# Matrix factorization model
python models/export/export_mf.py \
  --checkpoint routellm/routellm_mf \
  --output models/triton/mf/1/model.onnx

# BERT router model
python models/export/export_bert.py \
  --checkpoint routellm/routellm_bert \
  --output models/triton/bert/1/model.onnx
```

Custom checkpoints via `--checkpoint <path-or-hf-id>`.
Skip validation with `--skip-validation`.

## Prebuilt images

```bash
docker pull ghcr.io/ideacrafterslabs/routellm-triton:latest
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  ghcr.io/ideacrafterslabs/routellm-triton:latest
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MF_CHECKPOINT` | `routellm/routellm_mf` | HF model ID or path (build arg) |
| `BERT_CHECKPOINT` | `routellm/routellm_bert` | HF model ID or path (build arg) |

Pass build args:
```bash
docker compose -f models/docker-compose.yml build \
  --build-arg MF_CHECKPOINT=my-org/custom-mf
```

## Directory layout

```
models/
  export/              # ONNX export scripts
    export_mf.py
    export_bert.py
    requirements.txt
  triton/              # Triton model repository
    mf/config.pbtxt
    mf/1/model.onnx    # (generated)
    bert/config.pbtxt
    bert/1/model.onnx  # (generated)
  Dockerfile.dev       # Multi-stage: export + serve
  docker-compose.yml   # Dev stack
```
