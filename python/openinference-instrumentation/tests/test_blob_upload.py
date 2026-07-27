import base64
from typing import Any, Dict, List, Optional

import pytest

from openinference.instrumentation import (
    Blob,
    BlobUploader,
    TraceConfig,
    _blob_upload,
    load_blob_uploader,
)
from openinference.instrumentation.config import (
    OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
    OPENINFERENCE_BLOB_UPLOADER,
    REDACTED_VALUE,
)
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

PNG_BYTES = b"\x89PNG\r\n" + bytes(range(256)) * 40  # ~10KB of fake image data

PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()

INPUT_IMAGE_URL_KEY = (
    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.1."
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)
OUTPUT_IMAGE_URL_KEY = (
    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0."
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)


class InMemoryUploader:
    """A minimal BlobUploader implementation for tests — OpenInference ships
    no implementation of its own."""

    def __init__(self) -> None:
        self.store: Dict[str, bytes] = {}
        self.blobs: List[Blob] = []
        self.accepting = True

    def upload(self, blob: Blob) -> Optional[str]:
        if not self.accepting:
            return None
        uri = f"memory://{blob.content_sha256}"
        self.store[uri] = blob.data
        self.blobs.append(blob)
        return uri

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        self.accepting = False


def test_blob_derives_modality_and_digest() -> None:
    image_blob = Blob(data=PNG_BYTES, mime_type="image/png")
    assert image_blob.modality == "image"
    assert len(image_blob.content_sha256) == 64
    # The Blob contract is media-agnostic even though only images offload today.
    assert Blob(data=b"RIFF", mime_type="audio/wav").modality == "audio"
    assert Blob(data=b"%PDF-1.4", mime_type="application/pdf").modality == "document"
    assert Blob(data=b"", mime_type="video/mp4").modality == "video"


def test_in_memory_uploader_satisfies_protocol() -> None:
    assert isinstance(InMemoryUploader(), BlobUploader)


@pytest.mark.parametrize("key", [INPUT_IMAGE_URL_KEY, OUTPUT_IMAGE_URL_KEY])
def test_mask_uploads_oversized_image(key: str) -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    masked = config.mask(key, PNG_DATA_URI)
    assert isinstance(masked, str) and masked.startswith("memory://")
    assert uploader.store[masked] == PNG_BYTES


def test_mask_passes_blob_context_to_uploader() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI)
    assert len(uploader.blobs) == 1
    assert uploader.blobs[0].mime_type == "image/png"
    assert uploader.blobs[0].modality == "image"
    assert uploader.blobs[0].attribute_key == INPUT_IMAGE_URL_KEY


def test_mask_redacts_oversized_image_without_uploader() -> None:
    config = TraceConfig(base64_image_max_length=100)
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) == REDACTED_VALUE


def test_mask_redacts_when_uploader_rejects() -> None:
    uploader = InMemoryUploader()
    uploader.shutdown()  # upload() now returns None
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) == REDACTED_VALUE


def test_mask_keeps_small_image_inline() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(
        blob_uploader=uploader,
        base64_image_max_length=len(PNG_DATA_URI) + 1,
    )
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) == PNG_DATA_URI
    assert not uploader.store


def test_mask_keeps_external_image_url() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=10)
    url = "https://example.com/image.png"
    assert config.mask(INPUT_IMAGE_URL_KEY, url) == url
    assert not uploader.store


def test_hide_takes_precedence_over_upload() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(
        blob_uploader=uploader,
        hide_input_images=True,
        base64_image_max_length=100,
    )
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) is None
    # Hidden content must never reach storage.
    assert not uploader.store


def test_non_recording_span_skips_upload() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    from openinference.instrumentation import OITracer

    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    tracer = OITracer(
        TracerProvider(sampler=ALWAYS_OFF).get_tracer(__name__),
        config=config,
    )
    span = tracer.start_span("llm")
    span.set_attribute(INPUT_IMAGE_URL_KEY, PNG_DATA_URI)
    span.end()
    # No blob was uploaded for the sampled-out span.
    assert not uploader.store


class _FakeEntryPoint:
    def __init__(self, name: str, target: Any) -> None:
        self.name = name
        self._target = target

    def load(self) -> Any:
        return self._target


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, entry_points_list: List[_FakeEntryPoint]
) -> None:
    def fake_entry_points(*, group: str) -> List[_FakeEntryPoint]:
        assert group == _blob_upload.BLOB_UPLOADER_ENTRY_POINT_GROUP
        return entry_points_list

    monkeypatch.setattr(_blob_upload, "entry_points", fake_entry_points)


def test_load_blob_uploader_from_instance_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploader = InMemoryUploader()
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", uploader)])
    assert load_blob_uploader("mem") is uploader


def test_load_blob_uploader_instantiates_class_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    loaded = load_blob_uploader("mem")
    assert isinstance(loaded, InMemoryUploader)


def test_load_blob_uploader_unknown_name_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    assert load_blob_uploader("nope") is None


def test_load_blob_uploader_rejects_non_uploader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bad", object())])
    assert load_blob_uploader("bad") is None


def test_blob_uploader_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    monkeypatch.setenv(OPENINFERENCE_BLOB_UPLOADER, "mem")
    monkeypatch.setenv(OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH, "100")
    config = TraceConfig()
    assert isinstance(config.blob_uploader, InMemoryUploader)
    masked = config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI)
    assert isinstance(masked, str) and masked.startswith("memory://")


def test_blob_uploader_env_var_unset_leaves_none() -> None:
    config = TraceConfig()
    assert config.blob_uploader is None
