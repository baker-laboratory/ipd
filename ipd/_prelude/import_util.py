import importlib.util
import sys
from pathlib import Path
from typing import Any, List, TypeVar

T = TypeVar('T')

def cherry_pick_import(qualname: str, attribute: str = '', path: str = '') -> Any:
    """Import a specific attribute from a module without importing the entire package hierarchy.

    This function allows importing specific attributes from modules that might be involved
    in circular import dependencies. It bypasses Python's standard import mechanism by
    directly loading the module file.

    Args:
        qualname: The fully qualified name of the module (e.g., 'package.subpackage.module')
        attribute: The attribute to import from the module

    Returns:
        The requested attribute from the module

    Raises:
        ImportError: If the module cannot be found or loaded
        AttributeError: If the requested attribute doesn't exist in the module

    Examples:
        >>> import tempfile, os
        >>> # Setup a temporary module for testing
        >>> temp_dir = tempfile.mkdtemp()
        >>> os.makedirs(os.path.join(temp_dir, "test_pkg", "subpkg"), exist_ok=True)
        >>> with open(os.path.join(temp_dir, "test_pkg", "subpkg", "test_module.py"), "w") as f:
        ...     _ = f.write("test_value = 42\\ndef test_func(): return 'Hello'")
        >>> # Adjust sys.path to include our temp directory
        >>> import sys
        >>> old_path, old_cwd = sys.path.copy(), os.getcwd()
        >>> sys.path.insert(0, temp_dir)
        >>> os.chdir(temp_dir)
        >>> # Now we can test the function
        >>> value = cherry_pick_import("test_pkg.subpkg.test_module", "test_value", temp_dir)
        >>> value
        42
        >>> func = cherry_pick_import("test_pkg.subpkg.test_module", "test_func")
        >>> func()
        'Hello'
        >>> # Clean up
        >>> sys.path = old_path
        >>> os.chdir(old_cwd)
        >>> import shutil
        >>> shutil.rmtree(temp_dir)
    """
    if not attribute: qualname, attribute = qualname.rsplit('.', 1)
    if qualname not in sys.modules:
        module_name = qualname.split('.')[-1]
        path2 = path or Path(__file__).parent.parent.parent.resolve()
        module_path = Path(path2) / f'{qualname.replace(".","/")}.py'

        # Check if path exists, if not assume it's a directory with __init__.py
        if not module_path.exists():
            module_path = module_path.parent / "__init__.py"
            if not module_path.exists():
                raise ImportError(f"Could not find module at {module_path}")

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not hasattr(spec, 'loader') or not spec.loader:
            raise ImportError(f"Failed to create spec for {qualname} at {module_path}")

        loaded_module = importlib.util.module_from_spec(spec)
        sys.modules[qualname] = loaded_module
        spec.loader.exec_module(loaded_module)

    try:
        return getattr(sys.modules[qualname], attribute)
    except AttributeError as e:
        raise AttributeError(f"Cherry Picked Module '{qualname}' has no attribute '{attribute}'") from e

def cherry_pick_imports(qualname: str, attributes: str, path='') -> List[Any]:
    """Import multiple attributes from a module without importing the entire package hierarchy.

    This is a convenience function that calls cherry_pick_import for each attribute.

    Args:
        qualname: The fully qualified name of the module (e.g., 'package.subpackage.module')
        attributes: Space-separated string of attribute names to import

    Returns:
        A list of the requested attributes in the same order as specified

    Raises:
        ImportError: If the module cannot be found or loaded
        AttributeError: If any of the requested attributes don't exist in the module

    Examples:
        >>> import tempfile, os
        >>> # Setup a temporary module for testing
        >>> temp_dir = tempfile.mkdtemp()
        >>> os.makedirs(os.path.join(temp_dir, "test_pkg", "subpkg"), exist_ok=True)
        >>> with open(os.path.join(temp_dir, "test_pkg", "subpkg", "test_module.py"), "w") as f:
        ...     _ = f.write("test_value = 42\\ntest_string = 'Hello'\\ndef test_func(): return 'World'")
        >>> # Adjust sys.path to include our temp directory
        >>> import sys
        >>> old_path, old_cwd = sys.path.copy(), os.getcwd()
        >>> sys.path.insert(0, temp_dir)
        >>> os.chdir(temp_dir)
        >>> # Now we can test the function
        >>> val, func = cherry_pick_imports("test_pkg.subpkg.test_module", "test_value test_func", path=temp_dir)
        >>> val
        42
        >>> func()
        'Hello'
        >>> # Clean up
        >>> sys.path = old_path
        >>> os.chdir(old_cwd)
        >>> import shutil
        >>> shutil.rmtree(temp_dir)
    """
    attrlist = attributes.split()
    return [cherry_pick_import(qualname, attr, path) for attr in attrlist]
