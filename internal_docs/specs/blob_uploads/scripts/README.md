# Blob-upload demo script

A live demo of the [blob-upload design](../blob_uploads.md), driving a **real
auto-instrumented OpenAI vision call** — no hand-built spans. An oversized base64
image that today rides on the span as `__REDACTED__` is handed to a `BlobUploader` at
capture time; the span attribute records only the destination URI. The script imports
the **live** `openinference-instrumentation` package from this repo (a
`[tool.uv.sources]` path in its PEP 723 metadata), so it exercises the shipped
`Blob`/`BlobUploader`/`TraceConfig` code — no vendored copies. It prints the resulting
spans and exports them to Phoenix.

`image_blob_demo.py` makes one chat-completions vision request (text + a ~600 KB
base64 PNG), instrumented by openinference-instrumentation-openai. The same request
runs twice, changing only the `TraceConfig`: `message_content.image.image.url` =
`__REDACTED__` (default) vs a blob-store path (`TraceConfig(blob_uploader=...)`).

The demo store (`LocalBlobStore`, ~25 lines inlined in the script) is deliberately
simple: it satisfies the shipped `BlobUploader` protocol, writes content-addressed
files under `scripts/blob_store/` (gitignored), and returns the file's
**repo-root-relative path** as the URI, e.g.
`internal_docs/specs/blob_uploads/scripts/blob_store/3a7bd3….png`. Phoenix displays
the URI as an ordinary string attribute — resolving or rendering it is the backend's
responsibility. OpenInference ships no uploader implementation, which is exactly why
the demo carries its own.

## Prerequisites

```bash
# 1. Phoenix locally
pip install arize-phoenix
phoenix serve                    # http://localhost:6006

# 2. OpenAI API key (the script makes real vision calls)
export OPENAI_API_KEY=...

# 3. (optional) overrides
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export OPENAI_MODEL=gpt-4o-mini          # vision model
```

Only [`uv`](https://docs.astral.sh/uv/) is otherwise required — dependencies are PEP 723
inline metadata resolved into an ephemeral environment.

## Run

```bash
uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
```

The script prints the spans it produced (attribute by attribute, long values elided
with their true size) and exits; the blobs stay under `scripts/blob_store/`.

## What to look at in Phoenix (http://localhost:6006)

**Project `blob-upload-image-demo`** — one `ChatCompletion` LLM span per run, from
identical app code:

1. First run (default config): the span's input messages show the image part as
   `__REDACTED__` — today's released behavior for any input image whose base64
   exceeds 32,000 chars (the only alternative today is raising the budget and
   carrying ~884 KB of base64 on the span).
2. Second run (blob-upload config): the same attribute holds
   `internal_docs/…/blob_store/<sha>.png` — a short path instead of a redaction
   marker; the bytes are in that file, deduped by content hash.
3. `input.value` is small in both runs — the instrumentor's existing pre-pass strips
   the base64 image from the serialized request. Upgrading that redaction to a blob
   URI is future work (step 5 of the techspec's
   [media-type checklist](../blob_uploads.md#6-checklist-adding-offload-support-for-a-new-media-type)).

## Layout

```
scripts/
├── README.md
├── image_blob_demo.py   — the live TraceConfig(blob_uploader=...) mask() choke point
│                          (the techspec's shipped integration point), driven by a
│                          real auto-instrumented OpenAI vision call
└── blob_store/          — content-addressed demo storage (gitignored, created on first run)
```
