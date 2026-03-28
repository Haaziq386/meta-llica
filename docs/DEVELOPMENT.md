# Development Guide

## 1) Setup

```bash
cd incident_response_env
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
pip install pytest black
```

## 2) Run server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## 3) Run tests

```bash
pytest
```

## 4) Curl examples

### Health

```bash
curl -s http://localhost:7860/health
```

### List tasks

```bash
curl -s http://localhost:7860/tasks | jq
```

### Reset task

```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy_crashed_service"}' | jq
```

### Step action

```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"command":"query_logs","target":"payment-service"}' | jq
```

### State

```bash
curl -s http://localhost:7860/state | jq
```

### Grader

```bash
curl -s http://localhost:7860/grader | jq
```

### Baseline

```bash
curl -s -X POST http://localhost:7860/baseline \
  -H "Content-Type: application/json" \
  -d '{}' | jq
```

## 5) Docker build and run

```bash
docker build -f server/Dockerfile -t incident-env .
docker run --rm -p 7860:7860 incident-env
```

## 6) Deploy to Hugging Face Spaces (Docker)

1. Create a Docker Space.
2. Push project content with server/Dockerfile.
3. Confirm container listens on port 7860.
4. Verify /health returns {"status":"ok"}.

## 7) Common errors

- Import errors for openenv-core:
  - Use fallback interface in server/environment.py.
- /grader fails with "episode not complete":
  - Continue stepping until observation.done is true.
- /baseline returns fallback mode:
  - Set GOOGLE_API_KEY for live inference.
- Invalid target errors:
  - Check /tasks action schema and scenario service names.
