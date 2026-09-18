"""TraceConfig masking must cover automatic-function-calling tool calls."""

import pytest
from google.genai import types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_genai._wrappers import _SyncGenerateContent
from openinference.semconv.trace import (
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

TOOL_CALL_NAME_KEY = (
    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
    f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
)


def _response_with_one_automatic_function_call() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="65 degrees and foggy.")]
                ),
            )
        ],
        automatic_function_calling_history=[
            # the SDK seeds the history with the request contents first (models.py:6363)
            types.Content(role="user", parts=[types.Part.from_text(text="weather?")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="get_weather", args={"location": "San Francisco"}
                        )
                    )
                ],
            ),
        ],
    )


@pytest.mark.parametrize(
    "config, expect_tool_call",
    [
        pytest.param(TraceConfig(), True, id="output-messages-visible"),
        pytest.param(TraceConfig(hide_output_messages=True), False, id="hide-output-messages"),
    ],
)
def test_automatic_function_call_tool_call_follows_hide_output_messages(
    config: TraceConfig,
    expect_tool_call: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def generate_content(*, model: str, contents: object) -> types.GenerateContentResponse:
        return _response_with_one_automatic_function_call()

    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    _SyncGenerateContent(tracer=tracer)(
        generate_content,
        None,
        (),
        {"model": "gemini-2.0-flash", "contents": "weather?"},
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    tool_call_keys = [key for key in attributes if "tool_call" in key]
    if expect_tool_call:
        assert attributes[TOOL_CALL_NAME_KEY] == "get_weather"
    else:
        assert tool_call_keys == [], f"hidden output messages still carry {tool_call_keys}"
