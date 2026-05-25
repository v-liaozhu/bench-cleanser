# Vendored sources

This directory contains code vendored from internal Microsoft tooling. It is
**not** part of the public bench-cleanser API and is excluded from lint and
type-checks (`pyproject.toml [tool.ruff.lint.per-file-ignores]` and
`[tool.mypy] exclude`). Do not modify these files in place — re-vendor from
upstream instead.

## cloudgpt.py

CloudGPT OpenAI client helpers (Azure AD token provider, available-model
enum). Vendored from the internal CloudGPT Python helpers package. Used by
`bench_cleanser/llm_client.py:_create_async_client` to acquire Azure AD
tokens via the `az` CLI without requiring a static API key or PAT.

License: Microsoft internal use. Distribution outside Microsoft requires
re-vendoring with an appropriately-licensed equivalent.

If you fork bench-cleanser and don't have CloudGPT access: replace the
token provider in `llm_client.py` with `openai.AzureOpenAI(api_key=...)`
or any equivalent and delete this directory.
