# Contributing

Thanks for your interest in preorder4mlc.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

- `make lint` — ruff must pass.
- `make test` — the test suite must pass.
- Keep `pyproject.toml`, `.zenodo.json`, and `CITATION.cff` at the **same version**;
  CI fails if they diverge.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Releasing

Releases are cut by tagging `vX.Y.Z` (matching the version in the three files above);
the `Publish` workflow builds and uploads to PyPI via OIDC trusted publishing.
See `docs/REPRODUCING.md` for reproducing the paper results.
