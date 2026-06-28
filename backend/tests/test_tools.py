import pytest
from app.tools.registry import ManagedFileReader, SafeCalculator


def test_calculator_accepts_arithmetic():
    assert SafeCalculator().evaluate("(12 + 3) * 2") == 30


def test_calculator_rejects_function_calls():
    with pytest.raises(ValueError):
        SafeCalculator().evaluate("__import__('os').system('dir')")


@pytest.mark.asyncio
async def test_file_reader_blocks_parent_traversal(tmp_path):
    reader = ManagedFileReader(tmp_path / "uploads")
    reader.root.mkdir()
    with pytest.raises(ValueError):
        await reader.read("../secret.txt")
