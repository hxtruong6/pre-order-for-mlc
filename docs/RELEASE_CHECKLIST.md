# Release checklist (manual — not run by this plan)

Infrastructure is in place; publishing is a deliberate manual action.

## One-time setup
- [ ] Confirm the PyPI project name `preorder4mlc` is available
      (https://pypi.org/project/preorder4mlc/). If taken, pick a new name and update
      `pyproject.toml`, the badges, and `pip install` references.
- [ ] Create the project on TestPyPI and PyPI (or reserve via first upload).
- [ ] In the GitHub repo, create two **environments**: `testpypi` and `pypi`.
- [ ] Configure **OIDC trusted publishing** on PyPI/TestPyPI for this repo + environment
      (no API tokens stored). See https://docs.pypi.org/trusted-publishers/.
- [ ] Connect the repo to **Zenodo** (https://zenodo.org/account/settings/github/) so a
      GitHub Release mints a DOI. Add the DOI back into `CITATION.cff` and `.zenodo.json`
      once known, bumping the version in lockstep.

## Each release
- [ ] Bump version in `pyproject.toml`, `.zenodo.json`, `CITATION.cff` (all equal).
- [ ] Add a `CHANGELOG.md` entry.
- [ ] (Optional) Run the `Publish` workflow via **workflow_dispatch** to push to TestPyPI
      and verify the artifact installs from TestPyPI.
- [ ] Tag `vX.Y.Z` and push the tag → the `pypi` job publishes to PyPI.
- [ ] Create a GitHub Release for the tag → Zenodo archives it and mints a DOI.
