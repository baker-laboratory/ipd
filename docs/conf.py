import os
import sys
import inspect
from typing import Type

sys.path.insert(0, os.path.abspath('..'))
import importlib.metadata

# -- Project information -----------------------------------------------------
project = 'IPD'
author = 'Will Sheffler'
release = importlib.metadata.version('ipd')

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'sphinx_prompt',
    'sphinxemoji.sphinxemoji',
    'sphinx_last_updated_by_git',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',  # Supports Google/NumPy-style docstrings
    # 'sphinx.ext.viewcode',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    # 'sphinx_gallery.gen_gallery',  # For example galleries like Biotite
]

# -- HTML Theme Configuration ------------------------------------------------
pygments_style = 'sphinx'

html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 3,
    'show_nav_level': 3,
}
html_static_path = ['_static']
html_css_files = ['css/custom.css']

sphinx_gallery_conf = {
    'examples_dirs': 'examples/gallery',  # Source directory
    'gallery_dirs': 'auto_examples',  # Output directory
}

# extensions.remove('sphinx_gallery.gen_gallery')

autosummary_generate = False

autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    # "undoc-members": True,
    "show-inheritance": True,
    # "private-members": True,
    # "special-members": '__init__ __getitem__ __getattr__',
    "member-order": 'bysource',
    "exclude-members": "__weakref__",
}

intersphinx_mapping = {
    'biotite': ('https://www.biotite-python.org/latest', None),
    'numpy': ('https://numpy.org/doc/stable', None)
}

# def linkcode_resolve(domain, info):
#     if domain != 'py': return None
#     if not info['module']: return None
#     filename = info['module'].replace('.', '/')
#     urlfile = "https://github.com/baker-laboratory/ipd/blob/main/%s.py" % filename
#     url = linkcode_resolve_impl(domain, info)
#     print(f"Linkcode URL: {url}")  # For debugging purposes
#     assert url.startswith(urlfile)
#     return url

code_url = "https://github.com/baker-laboratory/ipd/blob/main/"

def linkcode_resolve(domain, info):
    # Non-linkable objects from the starter kit in the tutorial.
    if domain == "js" or info["module"] == "connect4": return
    assert domain == "py", "expected only Python objects"
    mod = importlib.import_module(info["module"])
    if "." not in info["fullname"]:
        obj = getattr(mod, info["fullname"])
        obj = inspect.unwrap(obj)
    else:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        obj = inspect.unwrap(obj)
        try:
            newobj = getattr(obj, attrname)
            newobj = inspect.unwrap(newobj)
            lines = inspect.getsourcelines(newobj)
            obj = newobj
        except AttributeError:
            return None
        except (OSError, TypeError):
            pass
    file = info['module'].replace('.', '/') + ".py"
    try:
        lines = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return f"{code_url}/{file}"
    start, end = lines[1], lines[1] + len(lines[0]) - 1
    return f"{code_url}/{file}#L{start}-L{end}"

# def skip_dataclass_field(app, what, name, obj, skip, options):
#     if what == "data":
#         cls = options.get("cls")
#         if cls and hasattr(cls, "__dataclass_fields__"):
#             if name in cls.__dataclass_fields__:
#                 return True
#     return skip

# def setup(app):
#     app.connect("autodoc-skip-member", skip_dataclass_field)
