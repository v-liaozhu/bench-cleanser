"""Test bootstrap.

Ensures ``pytest-asyncio`` is registered even in environments that disable
plugin autoloading (``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` set). Tests must not
touch the network — async tests use in-test fakes for the LLM client.
"""

from __future__ import annotations

pytest_plugins = ("pytest_asyncio",)
