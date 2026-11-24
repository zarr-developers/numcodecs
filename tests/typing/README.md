# Static type tests

The files here check numcodecs' type annotations against its runtime behavior. Each test
pairs `typing.assert_type`, which mypy and pyright check, with a runtime assertion, which
pytest checks. An annotation that type checkers accept but the runtime contradicts makes one
of the two fail.

Run them from the repository root:

```bash
uv pip install --group typing
pytest tests/typing
mypy --config-file tests/typing/mypy.ini tests/typing
pyright -p tests/typing
```

`mypy.ini` and `pyrightconfig.json` run both checkers in strict mode and read numcodecs from
`src/`. Errors inside numcodecs itself are not reported here: the repository's own mypy
configuration in `pyproject.toml` covers those.

`pyrightconfig.json` turns off pyright's three `reportUnknown*` rules, because most numcodecs
parameters are not annotated yet. Turn them back on once they are.

When you change an annotation in numcodecs, add or update an `assert_type` here that shows
what callers now get.
