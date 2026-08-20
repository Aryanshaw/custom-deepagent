
from typing import Any


class ToolMiddleware:

    def after_tool_call(self , name: str , args: dict[str , Any] , result: Any) -> Any:
        return result

    def around_tool_call(self, name: str , args: dict[str , Any] , result: Any) -> Any:
        pass

    def before_tool_call(self, name: str , args: dict[str , Any] , result: Any) -> Any:
        pass