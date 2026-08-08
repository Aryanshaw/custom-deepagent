from app.agent.tool_registry import tool


@tool
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
