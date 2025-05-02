import inspect
from functools import wraps
from typing import Optional, get_type_hints, Union


class Config:
    """Hierarchical configuration system."""

    def __init__(self, parent=None, **kwargs):
        self.parent = parent
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        elif self.parent:
            return self.parent.get(key, default)
        return default

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if self.parent:
            return getattr(self.parent, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' has no attribute '{name}'")


# Global default configuration
default_config = Config(timeout=30, retries=3, log_level='INFO')


def with_config(config_param_name='config', default_config=default_config):
    """
    Decorator that automatically applies configuration values to function parameters.

    Args:
        config_param_name: The name of the parameter that will receive the config object
        default_config: The default configuration to use if none is provided
    """

    def decorator(func):
        # Get the signature of the function
        sig = inspect.signature(func)
        parameters = sig.parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the config object from kwargs or use default
            config = kwargs.get(config_param_name, default_config)

            # Create a copy of kwargs to avoid modifying the original
            new_kwargs = kwargs.copy()

            # Iterate through function parameters
            for param_name, param in parameters.items():
                # Skip *args, **kwargs, and the config parameter itself
                if (param.kind == param.VAR_POSITIONAL
                        or param.kind == param.VAR_KEYWORD
                        or param_name == config_param_name):
                    continue

                # Skip parameters that are already provided in args or kwargs
                if param_name in kwargs:
                    continue

                # Skip parameters without default values (required params)
                if param.default is param.empty:
                    continue

                # If parameter has a default value of None, try to get it from config
                if param.default is None:
                    config_value = config.get(param_name)
                    if config_value is not None:
                        new_kwargs[param_name] = config_value

            return func(*args, **new_kwargs)

        return wrapper

    return decorator


# Example usage
@with_config(default_config=Config(timeout=10, retries=5, db_host='localhost'))
def fetch_data(url, *, timeout=None, retries=None, config=None):
    """Fetch data from a URL with configurable parameters."""
    print(f'Fetching {url} with timeout={timeout}, retries={retries}')
    # Implementation...


# More advanced version that considers type hints
def typed_with_config(config_param_name='config',
                      default_config=default_config):
    """
    Version of with_config that respects type hints when applying configuration.
    """

    def decorator(func):
        # Get the signature of the function
        sig = inspect.signature(func)
        parameters = sig.parameters

        # Get type hints
        type_hints = get_type_hints(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the config object from kwargs or use default
            config = kwargs.get(config_param_name, default_config)

            # Create a copy of kwargs to avoid modifying the original
            new_kwargs = kwargs.copy()

            # Iterate through function parameters
            for param_name, param in parameters.items():
                # Skip *args, **kwargs, and the config parameter itself
                if (param.kind == param.VAR_POSITIONAL
                        or param.kind == param.VAR_KEYWORD
                        or param_name == config_param_name):
                    continue

                # Skip parameters that are already provided in args or kwargs
                if param_name in kwargs:
                    continue

                # Skip parameters without default values (required params)
                if param.default is param.empty:
                    continue

                # If parameter has a default value of None, try to get it from config
                if param.default is None:
                    config_value = config.get(param_name)
                    if config_value is not None:
                        # If we have type hints, check if the config value matches
                        if param_name in type_hints:
                            expected_type = type_hints[param_name]
                            # Handle Optional types
                            if hasattr(expected_type, '__origin__'
                                       ) and expected_type.__origin__ is Union:
                                if type(None) in expected_type.__args__:
                                    valid_types = [
                                        t for t in expected_type.__args__
                                        if t is not type(None)
                                    ]
                                    if len(valid_types) == 1 and isinstance(
                                            config_value, valid_types[0]):
                                        new_kwargs[param_name] = config_value
                                    elif len(valid_types) > 1 and any(
                                            isinstance(config_value, t)
                                            for t in valid_types):
                                        new_kwargs[param_name] = config_value
                            # Simple type check
                            elif isinstance(config_value, expected_type):
                                new_kwargs[param_name] = config_value
                        else:
                            # No type hints, just use the value
                            new_kwargs[param_name] = config_value

            return func(*args, **new_kwargs)

        return wrapper

    return decorator


# Example with typed version
@typed_with_config()
def process_item(item_id: str,
                 *,
                 timeout: Optional[int] = None,
                 retries: Optional[int] = None,
                 config=None):
    """Process an item with type-checked configuration parameters."""
    print(f'Processing {item_id} with timeout={timeout}, retries={retries}')
    # Implementation...
