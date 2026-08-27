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

## Configure the server

Edit the `server` section in `config/config.yaml`:

```yaml
server:
  domain: 127.0.0.1
  port: 8000
  reload: true
```

Use `127.0.0.1` for local-only access or `0.0.0.0` when serving through a network or public domain. A public domain still needs DNS and, normally, a reverse proxy pointing to this configured port.

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
