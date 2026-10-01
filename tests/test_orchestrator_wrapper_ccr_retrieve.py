"""Regression tests: a headroom_retrieve result delivered through an orchestrator
wrapper must keep the ccr_retrieve exemption (#3563).

OpenCode V2 Code Mode runs every MCP call inside its built-in ``execute`` tool,
and Codex code mode sends calls as ``exec`` / ``functions.exec`` custom tool
calls whose payload is JavaScript. On the wire the tool name is the wrapper --
the inner name never reaches the proxy -- so the exemption that keys on the tool
name missed, SmartCrusher re-offloaded the retrieved bytes into a fresh
``<<ccr:hash>>`` marker, and the model could never redeem it (the same
unresolvable retrieval loop class as #1077 / #2698).

The fix resolves such a call to the retrieval tool's own name when -- and only
when -- the call's own payload invokes ``headroom_retrieve``. The wrapper is
never exempted as a whole, and the result content is never consulted (an
``original_content`` property is user-controllable data, not recovery proof).
"""

from __future__ import annotations

import json

from headroom.config import unwrap_tool_call
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


def _get_tokenizer():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


def _big_retrieve_json() -> str:
    """The JSON object the retrieval tool returns: big enough to clear the
    compression threshold, shaped like the #3563 repro (``original_content``)."""
    rows = "\n".join(
        f"Row {i:03d}: synthetic recovery check; status=healthy; payload=harmless sample."
        for i in range(90)
    )
    return json.dumps({"original_content": rows})


def _orchestrator_messages(wrapper_name: str, code: str, content: str) -> list[dict]:
    """OpenAI-shape conversation: an outer wrapper call, then its tool result."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_ccr_exec",
                    "type": "function",
                    "function": {"name": wrapper_name, "arguments": json.dumps({"code": code})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_ccr_exec", "content": content},
    ]


RETRIEVE_CODE = (
    'const r = await tools.headroom.headroom_retrieve({hash: "abc123def456"});\nreturn r;'
)
NON_RETRIEVE_CODE = 'const r = await tools.bash.bash({command: "ls"});\nreturn r;'


class TestOrchestratorUnwrap:
    """Unit coverage for the identity derivation itself."""

    def test_opencode_execute_wrapper_resolves_to_retrieve(self):
        name, _args = unwrap_tool_call("execute", json.dumps({"code": RETRIEVE_CODE}))
        assert name == "headroom_retrieve"

    def test_qualified_mcp_name_inside_script_resolves(self):
        code = 'const r = await tools.mcp__headroom__headroom_retrieve({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_bracket_access_inside_script_resolves(self):
        code = 'const r = await tools["headroom_retrieve"]({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_execute_wrapper_without_retrieve_call_keeps_name(self):
        assert unwrap_tool_call("execute", json.dumps({"code": NON_RETRIEVE_CODE}))[0] == "execute"

    def test_non_wrapper_mentioning_retrieve_keeps_name(self):
        # A non-orchestrator tool whose arguments merely name the tool is not
        # recovery output; only recognized wrappers resolve.
        args = '{"command": "grep -n headroom_retrieve( README.md"}'
        assert unwrap_tool_call("bash", args)[0] == "bash"

    def test_wrapper_payload_mentioning_retrieve_without_a_call_keeps_name(self):
        assert unwrap_tool_call("execute", json.dumps({"code": "// see headroom_retrieve"}))[0] == (
            "execute"
        )


class TestOrchestratorWrapperCcrRetrieveExemption:
    def test_execute_wrapper_retrieve_result_not_recompressed(self):
        """OpenCode V2 Code Mode: the execute wrapper's retrieval result must
        pass through verbatim instead of being re-offloaded."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _orchestrator_messages("execute", RETRIEVE_CODE, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"] == content, (
            "headroom_retrieve result behind the execute wrapper was recompressed "
            "(unresolvable retrieval loop, #3563)"
        )
        assert "<<ccr:" not in tool_msg["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_functions_exec_wrapper_retrieve_result_not_recompressed(self):
        """Codex code-mode shape: same guarantee under `functions.exec`."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        code = 'const r = await tools.mcp__headroom__headroom_retrieve({hash: "abc"});\nreturn r;'
        messages = _orchestrator_messages("functions.exec", code, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"] == content
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_execute_wrapper_without_retrieve_call_still_compressed(self):
        """The exemption stays narrow: an execute call that does not invoke the
        retrieval tool has ordinary, compressible output."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _orchestrator_messages("execute", NON_RETRIEVE_CODE, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert "router:excluded:ccr_retrieve" not in result.transforms_applied
        assert tool_msg["content"] != content or result.tokens_after < result.tokens_before
