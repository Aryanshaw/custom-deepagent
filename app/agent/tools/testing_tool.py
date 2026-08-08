from app.agent.tool_registry import tool
from app.config.logger import logger


@tool
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    logger.info(f"Tool call: add_two_numbers with args: {a}, {b}")
    result = a + b
    logger.info(f"Tool call: add_two_numbers with result: {result}")
    return result
