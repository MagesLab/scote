{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% set home = scote_homes[fullname] -%}
{% if home.modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: scote-module.rst
   :recursive:
{% for item in home.modules %}
   {{ item.split('.')[-1] }}
{%- endfor %}
{% endif %}
{% if home.functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
   :template: scote-function.rst
   :nosignatures:
{% for item in home.functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% if home.classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: scote-class.rst
   :nosignatures:
{% for item in home.classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% if home.data %}
.. rubric:: Constants
{% for item in home.data %}
.. autodata:: {{ fullname }}.{{ item }}
   :no-value:
{% endfor %}
{% endif %}
{% if home.reexports %}
.. rubric:: Also available here

Documented under their main name:
{% for name, target in home.reexports %}
- ``{{ fullname }}.{{ name }}``: :py:obj:`{{ target }}`
{%- endfor %}
{% endif %}
