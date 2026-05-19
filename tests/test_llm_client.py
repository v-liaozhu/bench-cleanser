"""Tests for bench_cleanser.llm_client JSON extraction and cache key determinism."""

import pytest

from bench_cleanser.llm_client import LLMClient


class TestExtractJson:
    """Test the static _extract_json method."""

    def test_clean_json(self):
        result = LLMClient._extract_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_fenced_json(self):
        text = '```json\n{"key": "value"}\n```'
        result = LLMClient._extract_json(text)
        assert result == {"key": "value"}

    def test_fenced_no_lang_tag(self):
        text = 'Some text\n```\n{"a": 1}\n```\nMore text'
        result = LLMClient._extract_json(text)
        assert result == {"a": 1}

    def test_brace_fallback(self):
        text = 'Here is the result: {"verdict": "CLEAN"} done.'
        result = LLMClient._extract_json(text)
        assert result == {"verdict": "CLEAN"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            LLMClient._extract_json("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            LLMClient._extract_json("")

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = LLMClient._extract_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}, "flag": True}


class TestCacheKeyDeterminism:
    """Verify cache keys are deterministic for the same inputs."""

    def test_same_input_same_key(self):
        from bench_cleanser.cache import ResponseCache

        key1 = ResponseCache.make_key("sys", "user", "model-1")
        key2 = ResponseCache.make_key("sys", "user", "model-1")
        assert key1 == key2

    def test_different_input_different_key(self):
        from bench_cleanser.cache import ResponseCache

        key1 = ResponseCache.make_key("sys", "user_a", "model-1")
        key2 = ResponseCache.make_key("sys", "user_b", "model-1")
        assert key1 != key2

    def test_different_model_different_key(self):
        from bench_cleanser.cache import ResponseCache

        key1 = ResponseCache.make_key("sys", "user", "model-1")
        key2 = ResponseCache.make_key("sys", "user", "model-2")
        assert key1 != key2

    def test_key_is_hex_string(self):
        from bench_cleanser.cache import ResponseCache

        key = ResponseCache.make_key("a", "b", "c")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in key)
