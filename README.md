# RAG API

A small **retrieval-augmented generation (RAG)** HTTP API built with **FastAPI**, **ChromaDB** (persistent vector store), and **Ollama** for local LLM inference. Documents are embedded and retrieved by semantic similarity; the top match is passed to the model as context for the answer.

## Features

- **POST `/query`** — Retrieve relevant context from ChromaDB and generate an answer with Ollama (`tinyllama` by default).
- **POST `/add`** — Append plain text to the knowledge base with a generated document ID.
- **`embed.py`** — One-off script to load `k8s.txt` into the `docs` collection (used by the Docker image at build time).

## Prerequisites

- **Python** 3.11+ (Dockerfile uses 3.13)
- **Ollama** installed and running, with the **`tinyllama`** model pulled:

  ```bash
  ollama pull tinyllama
  ```

- Dependencies (also installed in Docker):

  ```bash
  pip install chromadb ollama fastapi uvicorn
  ```

## Local development

1. Start Ollama (default: `http://localhost:11434`).

2. Seed the database (optional; creates `./db` and loads sample content from `k8s.txt`):

   ```bash
   python embed.py
   ```

3. Run the API:

   ```bash
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```

4. Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Example requests

Query parameters are used for `q` and `text` in the current app:

```bash
curl -X POST "http://127.0.0.1:8000/query?q=What%20is%20Kubernetes?"
```

```bash
curl -X POST "http://127.0.0.1:8000/add?text=Your%20knowledge%20snippet%20here."
```

Chroma data is stored under **`./db`** (gitignored if you add it to `.gitignore`).

## Docker

Build and run (Ollama must be reachable from the container; on Docker Desktop you can point the client at the host):

```bash
docker build -t rag-app .
docker run --rm -p 8000:8000 -e OLLAMA_HOST=host.docker.internal:11434 rag-app
```

The image runs `embed.py` during build so the baked-in `k8s.txt` content is in the image’s Chroma volume layer; for a writable persistent store, mount a volume over `/app/db` if you extend the setup.

## Kubernetes

Manifests in this repo:

| File | Purpose |
|------|---------|
| `deployment.yaml` | Deployment `rag-app-deployment`, port 8000, `OLLAMA_HOST=host.docker.internal:11434` for Ollama on the host |
| `service.yaml` | NodePort service `rag-app-service` on port 8000 |

Use a cluster where `host.docker.internal` (or your chosen `OLLAMA_HOST`) resolves to your Ollama instance, or adjust env vars and image names (`rag-app` with `imagePullPolicy: Never` assumes the image is loaded locally).

## Project layout

| Path | Role |
|------|------|
| `app.py` | FastAPI app, Chroma collection `docs`, `/query` and `/add` |
| `embed.py` | Loads `k8s.txt` into Chroma |
| `k8s.txt` | Sample knowledge file for embedding |
| `Dockerfile` | Production-style image with uvicorn |
| `deployment.yaml` / `service.yaml` | Kubernetes resources |

## Notes

- **`/query`** returns up to **one** retrieved chunk (`n_results=1`). Increase `n_results` in `app.py` if you want richer context.
- Change the Ollama **`model`** name in `app.py` if you use a different pulled model.

## License

Add a license file if you publish this repository.
