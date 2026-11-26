"""Regression tests for region-prefixed Bedrock inference-profile model IDs.

Cross-region inference profiles prefix the provider with a region code, e.g.
``us.meta.llama4-maverick-17b-instruct-v1:0``. The vendor must be derived from
any dot-separated segment so ``llm.output.value`` is populated exactly as it
is for the plain ``meta.*`` form.
"""

from typing import Any, Dict

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.bedrock.utils._extract_invoke_model_attributes import (
    _set_model_name_attributes,
)
from openinference.semconv.trace import SpanAttributes

OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE


@pytest.mark.parametrize(
    "model_id,response_body",
    [
        # Plain provider prefix (control) — must keep working.
        ("meta.llama3-8b-instruct-v1:0", {"generation": "hi"}),
        # Region-prefixed cross-region inference profiles.
        ("us.meta.llama4-maverick-17b-instruct-v1:0", {"generation": "hi"}),
        ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", {"completion": "hi"}),
        ("eu.meta.llama3-3-70b-instruct-v1:0", {"generation": "hi"}),
        ("apac.anthropic.claude-3-sonnet-20240229-v1:0", {"completion": "hi"}),
    ],
    ids=[
        "plain-meta-control",
        "us-meta",
        "us-anthropic",
        "eu-meta",
        "apac-anthropic",
    ],
)
def test_region_prefixed_model_id_sets_output_value(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    model_id: str,
    response_body: Dict[str, Any],
) -> None:
    span = tracer_provider.get_tracer(__name__).start_span("d270.region.vendor")
    _set_model_name_attributes(span, response_body, {"modelId": model_id})
    span.end()

    (finished,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(finished.attributes or {})
    assert attributes[OUTPUT_VALUE] == "hi"
