"""Sphinx helpers of the scote documentation.

- The public modules are ``PUBLIC_MODULES``; any other module of the package whose name has no component with a
  leading underscore is reported with a warning, so that it is made private or added to the API reference.
- One documented home per public object: an object exported by several public modules gets its page under the
  shallowest public module whose ``__all__`` lists it.
  :func:`api_homes` computes these homes for the autosummary templates (``_templates``), and references to any other
  path of the object resolve to its home.
- Inline code in Markdown that names a public object (``scote.io.read_bins``, or an unambiguous short name such as
  ``collapse_tensor``; also in the pages included from the README) becomes a link to its page.
- Markdown links to a page of the published documentation (``DOCS_URL``), as the README has them, point to the page of
  the build itself.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

from docutils import nodes
from sphinx.transforms import SphinxTransform
from sphinx.util import logging

logger = logging.getLogger(__name__)

PACKAGE = "scote"
DOCS_URL = "https://mageslab.github.io/scote"
PUBLIC_MODULES = ("scote", "scote.io", "scote.uncertainty")
_CODE_NAME = re.compile(rf"^({PACKAGE}(?:\.[A-Za-z_]\w*)*|[A-Za-z_]\w*)(?:\(.*\))?$")


# ---------------------------------------------------------------------------------------------------------------------
# one home per public object


def _public_modules(package: str) -> list[str]:
    """Names of the public modules of a package (``PUBLIC_MODULES``), shallowest first."""
    names = [m for m in PUBLIC_MODULES if m == package or m.startswith(package + ".")]
    return sorted(names, key=lambda n: (n.count("."), n))


def _unlisted_modules(package: str) -> list[str]:
    """Modules of a package that are not in ``PUBLIC_MODULES`` although no component of their name starts with an
    underscore."""
    pkg = importlib.import_module(package)
    return [info.name for info in pkgutil.walk_packages(pkg.__path__, package + ".")
            if info.name not in PUBLIC_MODULES and not any(part.startswith("_") for part in info.name.split("."))]


def _exported(module) -> list[str]:
    """Public names of a module: its ``__all__``, otherwise the public names defined in it."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n, o in vars(module).items()
                 if not n.startswith("_") and getattr(o, "__module__", None) == module.__name__]
    return [n for n in names if not n.startswith("_")]


def api_homes(package: str = PACKAGE) -> tuple[dict, dict]:
    """The documented home of every public object.

    Returns
    -------
    homes : dict
        Module name -> ``{"modules", "functions", "classes", "data", "reexports"}``: its public submodules (full
        names), the names documented on its page, and ``(name, home)`` of the objects it exports that are documented
        elsewhere.
    aliases : dict
        Every public path of an object (``module.name``) -> the full name of its home.
    """
    modules = _public_modules(package)
    homes = {m: {"modules": [], "functions": [], "classes": [], "data": [], "reexports": []} for m in modules}
    for m in modules:
        parent = m.rpartition(".")[0]
        if parent in homes:
            homes[parent]["modules"].append(m)
    home_of = {}                                             # id(object) -> full name of its home
    for m in modules:
        mod = importlib.import_module(m)
        for name in _exported(mod):
            obj = getattr(mod, name, None)
            if obj is None or inspect.ismodule(obj):
                continue
            if inspect.isclass(obj) or callable(obj):
                if id(obj) in home_of:
                    homes[m]["reexports"].append((name, home_of[id(obj)]))
                    continue
                home_of[id(obj)] = f"{m}.{name}"
                homes[m]["classes" if inspect.isclass(obj) else "functions"].append(name)
            else:
                homes[m]["data"].append(name)
    aliases = {}
    for m in modules:
        mod = importlib.import_module(m)
        for name, obj in vars(mod).items():
            if not name.startswith("_") and id(obj) in home_of:
                aliases[f"{m}.{name}"] = home_of[id(obj)]
    for m in modules:                                        # private modules that define public objects
        for name in ("functions", "classes"):
            for obj_name in homes[m][name]:
                obj = getattr(importlib.import_module(m), obj_name)
                defined = getattr(obj, "__module__", None)
                if defined:
                    aliases.setdefault(f"{defined}.{obj_name}", f"{m}.{obj_name}")
    return homes, aliases


def _home(target: str, aliases: dict) -> str | None:
    """The home of a dotted path, or of the object one of whose attributes it names (``Class.method``)."""
    parts = target.split(".")
    for k in range(len(parts), 0, -1):
        head = ".".join(parts[:k])
        if head in aliases:
            return ".".join([aliases[head]] + parts[k:])
    return None


# abbreviations of the annotations in signatures (``np.ndarray``) -> the names of the intersphinx inventories
_ABBREVIATIONS = {"np.": "numpy.", "sp.": "scipy.sparse.", "pd.": "pandas."}


def _missing_reference(app, env, node, contnode):
    """Resolve a Python reference to another path of a documented object (or a name relative to the current module),
    or an abbreviated module name of a dependency (``np.ndarray``) through intersphinx."""
    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"]
    for short, full in _ABBREVIATIONS.items():
        if target.startswith(short):
            from sphinx.ext import intersphinx

            new = node.deepcopy()
            new["reftarget"] = full + target[len(short):]
            return intersphinx.missing_reference(app, env, new, contnode)
    aliases = app.env.scote_aliases if hasattr(app.env, "scote_aliases") else {}
    candidates = [target]
    if node.get("py:module"):
        candidates.append(f"{node['py:module']}.{target}")
    domain = env.get_domain("py")
    for candidate in candidates:
        home = _home(candidate, aliases)
        if home and home != target:
            new = node.deepcopy()
            new["reftarget"] = home
            new["refspecific"] = False
            found = domain.resolve_any_xref(env, node["refdoc"], app.builder, home, new, contnode)
            if found:
                return found[0][1]
    return None


class CodeNameLinks(SphinxTransform):
    """Inline code that names a public object of the package becomes a reference to its documentation."""

    default_priority = 400

    def apply(self, **kwargs):
        from sphinx import addnodes

        aliases = getattr(self.env, "scote_aliases", {})
        for literal in list(self.document.findall(nodes.literal)):
            if isinstance(literal.parent, (nodes.reference, addnodes.pending_xref, nodes.literal_block)):
                continue
            match = _CODE_NAME.match(literal.astext())
            if not match:
                continue
            name = match.group(1)
            home = _home(name, aliases)
            if home is None and name in getattr(self.env, "scote_modules", ()):
                home = name
            if home is None and "." not in name:
                home = getattr(self.env, "scote_short_names", {}).get(name)
            if home is None:
                continue
            ref = addnodes.pending_xref("", refdomain="py", reftype="obj", reftarget=home, refexplicit=True,
                                        refwarn=False, refdoc=self.env.docname)
            literal.replace_self(ref)
            ref += literal


def _short_names(homes: dict) -> dict:
    """Unambiguous short names of the documented functions and classes (``read_visium_hd``, ``TwoPointResult``)
    -> their full names; names shorter than six characters or without an underscore or capital are left out, so that
    ordinary words in inline code stay plain code."""
    found = {}
    for module, home in homes.items():
        for name in home["functions"] + home["classes"]:
            found.setdefault(name, []).append(f"{module}.{name}")
    return {name: full[0] for name, full in found.items()
            if len(full) == 1 and len(name) >= 6 and ("_" in name or name != name.lower())}


def _note_api(app):
    for name in _unlisted_modules(PACKAGE):
        logger.warning(f"module {name} has a public name but is not in the API reference: make it private, or add it "
                       "to PUBLIC_MODULES in docs/_ext/scote_docs.py")
    homes, aliases = api_homes()
    app.env.scote_aliases = aliases
    app.env.scote_modules = set(homes)
    app.env.scote_short_names = _short_names(homes)


def _skip_namedtuple_fields(app, what, name, obj, skip, options):
    """Skip the field properties of a named tuple: its docstring documents the fields in its Attributes section."""
    if type(obj).__name__ == "_tuplegetter":
        return True
    return None


# ---------------------------------------------------------------------------------------------------------------------
# links to the published documentation -> pages of this build

_DOCS_LINK = re.compile(r"\]\(" + re.escape(DOCS_URL) + r"(?:/(?:index\.html)?|/([\w-]+)\.html)?(#[\w-]+)?\)")


def _local_docs_links(app, text: str) -> str:
    """``[text](DOCS_URL/page.html#anchor)`` -> ``[text](page.md#anchor)`` for the Markdown pages of the source
    directory (the site root -> ``index.md``); other links are left as they are."""
    def local(match):
        page = match.group(1) or "index"
        if not (app.srcdir / f"{page}.md").is_file():
            return match.group(0)
        return f"]({page}.md{match.group(2) or ''})"
    return _DOCS_LINK.sub(local, text)


def _local_links_page(app, docname, source):
    source[0] = _local_docs_links(app, source[0])


def _local_links_include(app, relative_path, parent_docname, content):
    content[0] = _local_docs_links(app, content[0])


def setup(app):
    app.connect("source-read", _local_links_page)
    app.connect("include-read", _local_links_include)
    app.connect("builder-inited", _note_api)
    app.connect("missing-reference", _missing_reference)
    app.connect("autodoc-skip-member", _skip_namedtuple_fields)
    app.add_transform(CodeNameLinks)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
