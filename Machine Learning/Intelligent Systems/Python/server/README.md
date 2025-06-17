# ML Server

## Installation

```bash
pipenv install
pipenv shell
```

#### Config File `.env`:

- `MODEL_DIR` --- folder with models
- `NUM_CORES` --- as it
- `MAX_LOADED_MODELS` as it

Execute the following from the project folder (where `.env` lies):

```bash
mkdir -p app/`MODEL_DIR`
```

---

## Launch the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Or use an existing Dockerfile

```bash
docker build -t ml-server .
docker run -d --name ml-server -p 8000:8000 -v "$(pwd)/.env:/app/.env" ml-server
```