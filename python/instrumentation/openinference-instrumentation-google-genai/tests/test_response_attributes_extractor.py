from typing import Any

import pytest
from google.genai import types

from openinference.instrumentation.google_genai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.semconv.trace import (
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)


@pytest.mark.parametrize(
    "usage_metadata, expected",
    [
        pytest.param(
            types.GenerateContentResponseUsageMetadata(
                total_token_count=110,
                prompt_token_count=10,
                prompt_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=7),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=3),
                ],
                candidates_token_count=80,
                candidates_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=11),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=69),
                ],
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.total": 110,
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 100,
                "llm.token_count.completion_details.reasoning": 20,
                "llm.token_count.prompt_details.audio": 7,
                "llm.token_count.completion_details.audio": 11,
            },
            id="all_fields",
        ),
    ],
)
def test_get_attributes_from_generate_content_usage(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
    expected: dict[str, Any],
) -> None:
    actual = dict(
        _ResponseAttributesExtractor()._get_attributes_from_generate_content_usage(usage_metadata)
    )
    assert actual == expected


@pytest.mark.parametrize(
    "usage_metadata, expected",
    [
        pytest.param(
            types.GenerateContentResponseUsageMetadata(
                cached_content_token_count=20,
                cache_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=14),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=6),
                ],
                total_token_count=110,
                prompt_token_count=30,
                prompt_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=7),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=3),
                ],
                candidates_token_count=80,
                candidates_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=11),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=69),
                ],
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.total": 130,
                "llm.token_count.prompt": 30,
                "llm.token_count.completion": 100,
                "llm.token_count.completion_details.reasoning": 20,
                "llm.token_count.prompt_details.cache_read": 20,
                "llm.token_count.prompt_details.audio": 7,
                "llm.token_count.completion_details.audio": 11,
            },
            id="all_fields",
        ),
    ],
)
def test_get_attributes_from_generate_content_usage_cached(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
    expected: dict[str, Any],
) -> None:
    actual = dict(
        _ResponseAttributesExtractor()._get_attributes_from_generate_content_usage(usage_metadata)
    )
    assert actual == expected


def test_automatic_function_calling_history_is_bucketed_into_output_messages() -> None:
    """AFC tool calls must sit inside an ``llm.output_messages.<i>`` bucket.

    The payload mirrors the recorded API response in
    ``tests/cassettes/test_instrumentation/test_generate_content_with_automatic_tool_calling.yaml``:
    Gemini sends ``functionCall`` with ``name``/``args`` only, no ``id``.
    """
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="It is 65 degrees and foggy.")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        automatic_function_calling_history=[
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
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="get_weather", response={"temperature": 65}
                        )
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="get_weather", args={"location": "New York"}
                        )
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="get_weather", response={"temperature": 72}
                        )
                    )
                ],
            ),
        ],
    )

    attributes = dict(_ResponseAttributesExtractor().get_attributes(response, {}))
    message_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1."
    assert attributes[f"{message_prefix}{MessageAttributes.MESSAGE_ROLE}"] == "model"
    tool_call_prefix = f"{message_prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
    assert (
        attributes[f"{tool_call_prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"]
        == "get_weather"
    )
    assert (
        attributes[f"{tool_call_prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
        == '{"location": "San Francisco"}'
    )
    # Each model entry gets its own bucket, and tool-call numbering restarts inside it.
    second_prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
    )
    assert (
        attributes[f"{second_prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_weather"
    )
    assert (
        attributes[f"{second_prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
        == '{"location": "New York"}'
    )
    # An un-prefixed bucket is invisible to every consumer that groups by message index.
    assert not [key for key in attributes if key.startswith("message.")]


def test_afc_history_entry_that_is_the_final_candidate_is_not_published_twice() -> None:
    """models.py:6356,6367 appends the same Content object the candidate already holds."""
    call_content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="get_weather", args={"location": "SF"})
            )
        ],
    )
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(index=0, content=call_content)],
        automatic_function_calling_history=[
            types.Content(role="user", parts=[types.Part.from_text(text="weather?")]),
            call_content,
        ],
    )

    attributes = dict(_ResponseAttributesExtractor().get_attributes(response, {}))
    tool_call_keys = [key for key in attributes if "tool_calls" in key]
    duplicate_bucket = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1."
    assert [key for key in tool_call_keys if key.startswith(duplicate_bucket)] == []
    assert (
        attributes[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
        ]
        == "get_weather"
    )


def test_afc_history_seeded_with_the_request_does_not_leak_earlier_turns() -> None:
    """models.py:6363 seeds the history with the request contents, which are inputs."""
    earlier_call = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="get_weather", args={"location": "SF"})
            )
        ],
    )
    request_contents = [
        types.Content(role="user", parts=[types.Part.from_text(text="weather?")]),
        earlier_call,
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="get_weather", response={"temperature": 65}
                    )
                )
            ],
        ),
    ]
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="72 in New York")]
                ),
            )
        ],
        automatic_function_calling_history=[
            *request_contents,
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="get_weather", args={"location": "NY"}
                        )
                    )
                ],
            ),
        ],
    )

    attributes = dict(
        _ResponseAttributesExtractor().get_attributes(response, {"contents": request_contents})
    )
    first = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
    assert (
        attributes[f"{first}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
        == '{"location": "NY"}'
    )
    assert not any(
        "San" in str(value) and key.startswith("llm.output_messages")
        for key, value in attributes.items()
    )


def test_afc_history_after_merged_request_parts_keeps_this_call_turn() -> None:
    """The seeded prefix is shorter than the caller's list when t_contents merges parts.

    ``_transformers.t_contents`` folds consecutive part-like items into one Content, so counting
    the caller's list would skip one entry too many and delete the call this invocation made.
    """
    request_parts = [
        types.Part.from_text(text="What's the weather like"),
        types.Part.from_text(text="in San Francisco?"),
    ]
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="65 degrees and foggy.")]
                ),
            )
        ],
        automatic_function_calling_history=[
            types.Content(role="user", parts=request_parts),
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

    attributes = dict(
        _ResponseAttributesExtractor().get_attributes(response, {"contents": request_parts})
    )
    tool_call_prefix = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
    )
    assert (
        attributes[f"{tool_call_prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"]
        == "get_weather"
    )
