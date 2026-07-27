# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.60.0",
#     "openinference-instrumentation-openai>=0.1.52",
#     "openinference-instrumentation",
#     "openinference-semantic-conventions>=0.1.30",
#     "opentelemetry-sdk>=1.42.0",
#     "opentelemetry-exporter-otlp-proto-http>=1.42.0",
#     "pillow>=10.0.0",
# ]
#
# [tool.uv.sources]
# openinference-instrumentation = { path = "../../../../python/openinference-instrumentation" }
# ///
"""Image path: a real auto-instrumented OpenAI vision call, redaction vs blob upload.

Runs the same chat-completions vision request twice through the released
openinference-instrumentation-openai auto-instrumentor. The app code never
changes — only the ``TraceConfig`` handed to the instrumentor does:

  run 1 — default config     the >32 KB base64 image is replaced with
                             ``__REDACTED__`` (today's released behavior).
  run 2 — blob-upload config ``TraceConfig(blob_uploader=...)`` — the same
                             attribute key records the blob store URI (a
                             repo-relative file path from the demo store).

The blob-upload pieces (``Blob``, ``BlobUploader``, the ``TraceConfig`` field
and mask policy) come from the **live** ``openinference-instrumentation``
package in this repo, resolved via the ``[tool.uv.sources]`` path above — this
script exercises the shipped code, not a local copy of it.

Prerequisites: OPENAI_API_KEY; a local ``phoenix serve`` (http://localhost:6006).
Run:  uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
"""

from __future__ import annotations

import base64
import os
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

from openai import OpenAI
from openinference.instrumentation import Blob, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import ImageAttributes, MessageContentAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image, ImageDraw

PROJECT_NAME = "blob-upload-image-demo"

_EXT_BY_MIME = {"image/png": ".png", "image/jpeg": ".jpg"}


class LocalBlobStore:
    """Mock ``BlobUploader``: content-addressed files in the local repo.

    OpenInference ships no uploader implementation — this ~25-line store
    satisfies the shipped ``BlobUploader`` protocol for the demo. It writes
    content-addressed files under ``scripts/blob_store/`` and returns the
    file's repo-root-relative path as the URI. Writes synchronously — fine for
    a demo; real implementations move bytes on a background worker.
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or Path(__file__).parent / "blob_store"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._repo_root = next(
            (p for p in [self.root_dir, *self.root_dir.resolve().parents] if (p / ".git").exists()),
            None,
        )

    def upload(self, blob: Blob) -> Optional[str]:
        name = blob.content_sha256[:20] + _EXT_BY_MIME.get(blob.mime_type, ".bin")
        path = self.root_dir / name
        if not path.exists():  # content-addressed dedup
            path.write_bytes(blob.data)
            print(f"[blob] stored {blob.modality} ({len(blob.data):,} B) → {path.name}")
        if self._repo_root is not None:
            return str(path.resolve().relative_to(self._repo_root))
        return path.resolve().as_posix()

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass  # synchronous mock — nothing pending


def make_demo_png() -> bytes:
    """A labeled banner over seeded RGB noise — noise is incompressible, so the
    PNG lands in the hundreds-of-KB range a real photo occupies, far over the
    32,000-char base64 budget TraceConfig allows an image today."""
    rng = random.Random(42)
    width, height = 640, 400
    img = Image.frombytes("RGB", (width, height), rng.randbytes(width * height * 3))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 56], fill=(16, 24, 48))
    draw.text((16, 12), "OpenInference blob-upload demo", fill=(240, 240, 255))
    draw.text((16, 32), "synthetic test pattern (seeded noise)", fill=(160, 170, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask_about_image(data_uri: str) -> str:
    """The real app: one chat-completions vision call."""
    response = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


IMAGE_URL_SUFFIX = (
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)


def print_spans(memory: InMemorySpanExporter, label: str, since: int) -> int:
    """Print each span from this run, attribute by attribute."""
    spans = memory.get_finished_spans()[since:]
    for span in spans:
        attributes = span.attributes or {}
        total = sum(len(k) + len(str(v)) for k, v in attributes.items())
        print(f"\n── {span.name} — {label}  ({len(attributes)} attrs, {total:,} B) ──")
        for key in sorted(attributes):
            text = str(attributes[key]).replace("\n", "\\n")
            if len(text) > 76:
                text = f"{text[:76]}… ({len(text):,} chars)"
            print(f"  {key} = {text}")
    return len(memory.get_finished_spans())


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set — this demo makes real vision calls.")

    png = make_demo_png()
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    print(
        f"generated image: {len(png):,} B PNG → {len(data_uri):,} chars as a data URI"
    )

    phoenix = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": PROJECT_NAME})
    )
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(f"{phoenix}/v1/traces"))
    )

    store = LocalBlobStore()
    seen = 0
    for label, config in [
        ("default config (image __REDACTED__)", TraceConfig()),
        ("blob upload (external URI)", TraceConfig(blob_uploader=store)),
    ]:
        OpenAIInstrumentor().instrument(tracer_provider=provider, config=config)
        answer = ask_about_image(data_uri)
        OpenAIInstrumentor().uninstrument()
        print(f"\n[{label}] model: {answer}")
        provider.force_flush()
        seen = print_spans(memory, label, seen)

    provider.shutdown()

    print(f"\nPhoenix: {phoenix}  → project {PROJECT_NAME!r}")
    print("Compare the two runs' ChatCompletion spans: message_content.image.image.url")
    print("is __REDACTED__ in the first and a repo-relative blob-store path in the")
    print("second — displaying/resolving that URI is the backend's concern.")


if __name__ == "__main__":
    main()
