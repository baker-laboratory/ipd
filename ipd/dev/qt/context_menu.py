import abc
from inspect import signature
from typing import Any, Callable

import pydantic

class MenuAction(pydantic.BaseModel):
    func: Callable[Any, None]  # type: ignore
    owner: bool = False
    item: bool = True

class ContextMenuMixin(abc.ABC):
    def _install_event_filter(self, parent):
        self.widget.installEventFilter(parent)  # type: ignore

    @abc.abstractmethod
    def _context_menu_items(self):
        'must return dict of MenuActions'

    @abc.abstractmethod
    def _get_from_item(self, item):
        'must return object represented by listitem'

    def _action_is_allowed(self, thing, action):
        return not action.owner or self.state.user in (thing.user.name, 'admin')  # type: ignore

    def context_menu(self, event):
        menu, thing = self.Qt.QtWidgets.QMenu(), None  # type: ignore
        if item := self.widget.itemAt(event.pos()):  # type: ignore
            thing = self._get_from_item(item)
            assert thing and not isinstance(thing, str), thing
            for name, action in self._context_menu_items().items():  # type: ignore
                if action.item:
                    menu.addAction(name).setEnabled(self._action_is_allowed(thing, action))
        for name, action in self._context_menu_items().items():  # type: ignore
            if not action.item:
                menu.addAction(name).setEnabled(self._action_is_allowed(thing, action))
        if selection := menu.exec_(event.globalPos()):
            func = self._context_menu_items()[selection.text()].func  # type: ignore
            params = signature(func).parameters
            if len(params) > 0: func(thing)
            else: func()
        return True
