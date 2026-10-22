# Building the documentation

The documentation is built with Sphinx from the Markdown pages of this directory and the docstrings of the package.
GitHub Actions builds it when a release is published and deploys it to https://mageslab.github.io/scote
(`.github/workflows/docs.yml`; Actions > Docs > Run workflow builds a given tag again).

## Build locally

From the root of a clone, install the package with the `docs` extra, then build:

```bash
pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html      # or: make -C docs html
```

Open `docs/_build/html/index.html` in a browser (the pages are static HTML), or serve them with
`python -m http.server -d docs/_build/html` and open http://localhost:8000 (`make -C docs serve` does both).

The build writes the HTML to `docs/_build/` and one API page per module, function, and class to `docs/generated/`;
neither is part of the repository, and `make -C docs clean` removes both.
