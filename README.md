# UP Infer

UP Infer is a small browser interface for uploading images and inspecting raw YOLO detection or segmentation results. It detects the model task automatically, draws boxes or masks, and reports the confidence of every predicted object.

## Run

The requested virtual environment is `/opt/personal/.personal-venv`.

```bash
cd /opt/personal/up_infer
/opt/personal/.personal-venv/bin/pip install -r requirements.txt
/opt/personal/.personal-venv/bin/python run.py
```

Open <http://127.0.0.1:8000>. API documentation is at <http://127.0.0.1:8000/docs>.

## Configure feedback

The ticket button in the top-right header forwards feedback to the centralized ticket service through the UP Infer backend. Configure the server environment before starting the app:

```bash
export TICKET_SERVICE_URL="https://ticket.agbropro.my.id"
export TICKET_SERVICE_API_KEY="your-central-ingestion-api-key"
```

The API key stays on the server and is never sent to the browser. Widget choices are mapped to the central categories as follows: `misc` to `general`, `bug` to `bug`, and `feature` to `feedback`. The original choice remains available as `metadata.feedback_type`.

## Configure the server

Edit the `server` section in `config/config.yaml`:

```yaml
server:
  domain: 127.0.0.1
  port: 8000
  reload: true
```

Use `127.0.0.1` for local-only access or `0.0.0.0` when serving through a network or public domain. A public domain still needs DNS and, normally, a reverse proxy pointing to this configured port.

## Configure GPU memory

The `inference` section controls GPU usage:

```yaml
inference:
  batch_size: 1
  image_size: 640
  device: auto
  half: true
  clear_cache: true
```

Keep `batch_size: 1` for predictable VRAM usage. Increase it only when throughput matters more than memory. Use `device: cpu` to disable CUDA, or a value such as `cuda:0` to select a GPU. The application keeps model weights cached for fast switching but allows only the most recently inferred model to retain a GPU predictor.

## Configure models

Edit `config/config.yaml` and add an entry for each Ultralytics-compatible model:

```yaml
models:
  - id: safety-detection
    name: Safety Detection
    path: /absolute/path/to/best.pt
```

Paths can be absolute, project-relative, or an Ultralytics model name such as `yolo11n.pt`. Named models are downloaded by Ultralytics on their first use. Model labels and the detection/segmentation task come directly from the loaded weights.

## Structure

```text
app/domain/          Typed response entities
app/application/     Configuration, model loading, and inference
app/interfaces/      FastAPI routes and upload validation
frontend/            Plain HTML, CSS, and JavaScript
config/config.yaml   Model and upload configuration
```
