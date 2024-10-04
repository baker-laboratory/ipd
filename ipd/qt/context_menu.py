import pydantic
import abc
from typing import Callable, Any

class MenuAction(pydantic.BaseModel):
    func: Callable[Any, None]
    owner: bool = False
    item: bool = True

class ContextMenuMixin(abc.ABC):
    @abc.abstractmethod
    def _context_menu_items(self):
        'must return dict of MenuActions'

    @abc.abstractmethod
    def get_from_item(self, item):
        'must return object represented by listitem'

    def action_is_allowed(self, thing, action):
        return not action.owner or self.state.user in (thing.user, 'admin')

    def context_menu(self, event):
        menu, thing = self.Qt.QtWidgets.QMenu(), None
        if item := self.widget.itemAt(event.pos()):
            thing = self.get_from_item(item)
            for name, action in self._context_menu_items().items():
                if action.item:
                    menu.addAction(name).setEnabled(self.action_is_allowed(thing, action))
        for name, action in self._context_menu_items().items():
            if not action.item:
                menu.addAction(name).setEnabled(self.action_is_allowed(thing, action))
        if selection := menu.exec_(event.globalPos()):
            try:
                self._context_menu_items()[selection.text()].func(thing)
            except TypeError:
                self._context_menu_items()[selection.text()].func()
        return True
