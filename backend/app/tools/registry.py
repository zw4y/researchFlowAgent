import ast
import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ToolCall, new_id
from app.schemas import ToolCallOut


class SearchDocumentsArgs(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    paper_ids: list[str] = Field(default_factory=list)


class AskPaperArgs(SearchDocumentsArgs):
    pass


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=10)


class QueryMetricsArgs(BaseModel):
    sql: str = Field(min_length=1, max_length=5000)


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class ReadFileArgs(BaseModel):
    relative_path: str = Field(min_length=1, max_length=500)


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


@dataclass(slots=True)
class ToolExecution:
    value: Any
    trace: ToolCallOut


class ToolRegistry:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def schemas(self) -> list[dict[str, Any]]:
        return [definition.schema() for definition in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any], run_id: str) -> ToolExecution:
        call_id = new_id()
        started = time.perf_counter()
        status: Literal["completed", "failed"] = "completed"
        result: Any = None
        summary: str | None = None
        error_message: str | None = None
        try:
            definition = self._tools[name]
            validated = definition.args_model.model_validate(arguments)
            result = await definition.handler(validated)
            summary = self._summarize(result)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)[:500]
        duration_ms = int((time.perf_counter() - started) * 1000)
        trace = ToolCallOut(
            id=call_id,
            name=name,
            arguments=arguments,
            status=status,
            duration_ms=duration_ms,
            result_summary=summary,
            error_message=error_message,
        )
        async with self.session_factory() as session:
            session.add(
                ToolCall(
                    id=call_id,
                    run_id=run_id,
                    name=name,
                    arguments=arguments,
                    status=status,
                    duration_ms=duration_ms,
                    result_summary=summary,
                    error_message=error_message,
                )
            )
            await session.commit()
        return ToolExecution(value=result, trace=trace)

    @staticmethod
    def _summarize(value: Any) -> str:
        if isinstance(value, list):
            return f"返回 {len(value)} 条结果"
        text = str(value)
        return text[:300]


class SafeCalculator:
    _operators: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    def evaluate(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return float(self._evaluate_node(tree.body))

    def _evaluate_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._evaluate_node(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in self._operators:
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            if abs(left) > 1e12 or abs(right) > 1e12:
                raise ValueError("数值过大")
            return self._operators[type(node.op)](left, right)
        raise ValueError("表达式包含不允许的语法")


class ManagedFileReader:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    async def read(self, relative_path: str) -> str:
        candidate = (self.root / relative_path).resolve()
        if self.root not in candidate.parents:
            raise ValueError("文件路径超出受管目录")
        if candidate.suffix.lower() not in {".txt", ".md", ".csv"}:
            raise ValueError("read_file 仅允许 txt、md、csv")
        return candidate.read_text(encoding="utf-8")[:50_000]
