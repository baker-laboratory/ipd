# basic_click_type_handlers.py
import pathlib
import click
import uuid
from evn.cli.click_type_handler import ClickTypeHandler, MetadataPolicy


class BasicStringHandler(ClickTypeHandler):
    __supported_types__ = {str: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        try:
            preprocessed = self.preprocess_value(value)
            # For string, we simply ensure it is a unicode string.
            return self.postprocess_value(str(preprocessed))
        except Exception as e:
            self.fail(f'BasicStringHandler conversion failed: {e}', param, ctx)


class BasicBoolHandler(ClickTypeHandler):
    __supported_types__ = {bool: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        val = self.preprocess_value(value)
        if isinstance(val, bool):
            result = val
        elif isinstance(val, str):
            lower = val.lower()
            if lower in ['true', '1', 'yes', 'on', 't']:
                result = True
            elif lower in ['false', '0', 'no', 'off', 'f']:
                result = False
            else:
                self.fail(f'BasicBoolHandler conversion failed for {value}',
                          param, ctx)
        else:
            self.fail(
                f'BasicBoolHandler conversion got unsupported type: {type(val)}',
                param, ctx)
        return self.postprocess_value(result)


class BasicUUIDHandler(ClickTypeHandler):
    __supported_types__ = {uuid.UUID: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        try:
            preprocessed = self.preprocess_value(value)
            result = uuid.UUID(preprocessed)
            return self.postprocess_value(result)
        except Exception as e:
            self.fail(f'BasicUUIDHandler conversion failed: {e}', param, ctx)


class BasicPathHandler(ClickTypeHandler):
    __supported_types__ = {pathlib.Path: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        try:
            preprocessed = self.preprocess_value(value)
            path_type = click.Path()
            result = path_type.convert(preprocessed, param, ctx)
            return self.postprocess_value(result)
        except Exception as e:
            self.fail(f'BasicPathHandler conversion failed: {e}', param, ctx)


class BasicChoiceHandler(ClickTypeHandler):
    __supported_types__ = {click.Choice: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        try:
            preprocessed = self.preprocess_value(value)
            # Assume that the handler instance has an attribute 'choices'
            if not hasattr(self, 'choices'):
                self.fail("BasicChoiceHandler missing 'choices' attribute",
                          param, ctx)
            choice_type = click.Choice(self.choices,
                                       case_sensitive=getattr(
                                           self, 'case_sensitive', True))
            result = choice_type.convert(preprocessed, param, ctx)
            return self.postprocess_value(result)
        except Exception as e:
            self.fail(f'BasicChoiceHandler conversion failed: {e}', param, ctx)


class BasicIntRangeHandler(ClickTypeHandler):
    __supported_types__ = {click.IntRange: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        try:
            preprocessed = self.preprocess_value(value)
            int_range = click.IntRange(
                min=getattr(self, 'min', None),
                max=getattr(self, 'max', None),
                min_open=getattr(self, 'min_open', False),
                max_open=getattr(self, 'max_open', False),
                clamp=getattr(self, 'clamp', False),
            )
            result = int_range.convert(preprocessed, param, ctx)
            return self.postprocess_value(result)
        except Exception as e:
            self.fail(f'BasicIntRangeHandler conversion failed: {e}', param,
                      ctx)


class BasicFloatRangeHandler(ClickTypeHandler):
    __supported_types__ = {click.FloatRange: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        try:
            preprocessed = self.preprocess_value(value)
            float_range = click.FloatRange(
                min=getattr(self, 'min', None),
                max=getattr(self, 'max', None),
                min_open=getattr(self, 'min_open', False),
                max_open=getattr(self, 'max_open', False),
                clamp=getattr(self, 'clamp', False),
            )
            result = float_range.convert(preprocessed, param, ctx)
            return self.postprocess_value(result)
        except Exception as e:
            self.fail(f'BasicFloatRangeHandler conversion failed: {e}', param,
                      ctx)


class BasicDateTimeHandler(ClickTypeHandler):
    __supported_types__ = {click.DateTime: MetadataPolicy.FORBID}
    __priority_bonus__ = 0

    def convert(self, value, param, ctx):
        try:
            preprocessed = self.preprocess_value(value)
            dt_type = click.DateTime(formats=getattr(self, 'formats', None))
            result = dt_type.convert(preprocessed, param, ctx)
            return self.postprocess_value(result)
        except Exception as e:
            self.fail(f'BasicDateTimeHandler conversion failed: {e}', param,
                      ctx)
