import uuid
from typing import Dict, List


class HookRegistry:
    def __init__(self):
        self._hooks: Dict[str, Dict] = {}

    def add(self, hook_name: str, code: str) -> str:
        hook_id = str(uuid.uuid4())[:8]
        self._hooks[hook_id] = {
            "id": hook_id,
            "hook_name": hook_name,
            "code": code,
        }
        return hook_id

    def remove(self, hook_id: str) -> bool:
        if hook_id in self._hooks:
            del self._hooks[hook_id]
            return True
        return False

    def list_hooks(self) -> List[Dict]:
        return list(self._hooks.values())

    def clear(self):
        self._hooks.clear()


hook_registry = HookRegistry()
