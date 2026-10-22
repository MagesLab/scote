# Build the documentation (see README.md): make -C docs html, make -C docs serve, make -C docs clean.
SPHINXBUILD ?= python -m sphinx
SPHINXOPTS  ?= -W --keep-going
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: html clean serve

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)" generated

serve: html
	python -m http.server -d "$(BUILDDIR)/html" 8000
