# API reference

The public modules are {mod}`scote` (the two-point functions, their result and backend options, and the thread
helpers), {mod}`scote.io` (readers of the vendor outputs, converters to the common data model and its readers, feature
selection), and {mod}`scote.uncertainty` (the closed-form tile block bootstrap and the delta-method statistics). Every
object has one page, under the module it is imported from. The command line `python -m scote` wraps the converters and
{func}`scote.twopoint` for workflows that run one step per process (`python -m scote --help`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: scote-module.rst
   :recursive:

   scote
```
