import csv
import io
from typing import Any

import sqlglot
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlglot import exp

from app.core.errors import AppError
from app.db.models import ExperimentMetric, Paper

ALLOWED_COLUMNS = {
    "paper_id",
    "experiment",
    "metric_name",
    "metric_value",
    "unit",
    "split",
    "notes",
}
ALLOWED_FUNCTIONS = {"avg", "count", "max", "min", "round", "sum"}


class SafeSQLValidator:
    def validate(self, sql: str) -> str:
        candidate = sql.strip()
        if not candidate or ";" in candidate or "--" in candidate or "/*" in candidate:
            raise AppError("SQL 包含不允许的分隔符或注释。", code="unsafe_sql")
        try:
            statements = sqlglot.parse(candidate, read="postgres")
        except sqlglot.errors.ParseError as exc:
            raise AppError("SQL 无法解析。", code="invalid_sql") from exc
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            raise AppError("只允许单条 SELECT 查询。", code="unsafe_sql")
        statement = statements[0]
        if statement.find(exp.Subquery) or statement.find(exp.Union):
            raise AppError("不允许子查询或集合查询。", code="unsafe_sql")
        tables = {table.name.lower() for table in statement.find_all(exp.Table)}
        if tables != {"experiment_metrics"}:
            raise AppError("只能查询 experiment_metrics 表。", code="unsafe_sql")
        columns = {column.name.lower() for column in statement.find_all(exp.Column)}
        if not columns.issubset(ALLOWED_COLUMNS):
            raise AppError("查询包含未授权字段。", code="unsafe_sql")
        if statement.find(exp.Star):
            raise AppError("请明确选择字段，不允许 SELECT *。", code="unsafe_sql")
        for function in statement.find_all(exp.Func):
            name = function.sql_name().lower()
            if name not in ALLOWED_FUNCTIONS:
                raise AppError(f"不允许 SQL 函数 {name}。", code="unsafe_sql")
        limit = statement.args.get("limit")
        if limit is None:
            statement = statement.limit(50)
        else:
            limit_expression = limit.expression
            if isinstance(limit_expression, exp.Literal) and int(limit_expression.this) > 50:
                statement.set("limit", exp.Limit(expression=exp.Literal.number(50)))
        return statement.sql(dialect="postgres")


class MetricService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory
        self.validator = SafeSQLValidator()

    async def import_csv(self, paper_id: str, content: bytes, *, replace: bool = True) -> int:
        async with self.session_factory() as session:
            if await session.get(Paper, paper_id) is None:
                raise AppError("论文不存在。", status_code=404, code="paper_not_found")
        try:
            decoded = content.decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(decoded)))
        except UnicodeDecodeError as exc:
            raise AppError("CSV 必须使用 UTF-8 编码。", code="invalid_csv") from exc
        required = {"experiment", "metric_name", "metric_value"}
        if not rows or not required.issubset(rows[0]):
            raise AppError(
                "CSV 必须包含 experiment、metric_name、metric_value 列。",
                code="invalid_csv",
            )
        metrics: list[ExperimentMetric] = []
        for index, row in enumerate(rows, start=2):
            try:
                value = float(row["metric_value"])
            except (TypeError, ValueError) as exc:
                raise AppError(
                    f"CSV 第 {index} 行 metric_value 不是数字。",
                    code="invalid_csv",
                ) from exc
            metrics.append(
                ExperimentMetric(
                    paper_id=paper_id,
                    experiment=row["experiment"].strip(),
                    metric_name=row["metric_name"].strip(),
                    metric_value=value,
                    unit=(row.get("unit") or "").strip() or None,
                    split=(row.get("split") or "").strip() or None,
                    notes=(row.get("notes") or "").strip() or None,
                )
            )
        async with self.session_factory() as session:
            if replace:
                await session.execute(
                    delete(ExperimentMetric).where(ExperimentMetric.paper_id == paper_id)
                )
            session.add_all(metrics)
            await session.commit()
        return len(metrics)

    async def execute_readonly(self, sql: str) -> list[dict[str, Any]]:
        validated = self.validator.validate(sql)
        async with self.session_factory() as session:
            if session.bind and session.bind.dialect.name == "postgresql":
                await session.execute(text("SET TRANSACTION READ ONLY"))
            result = await session.execute(text(validated))
            return [dict(row) for row in result.mappings().all()]

    async def list_for_paper(self, paper_id: str) -> list[ExperimentMetric]:
        async with self.session_factory() as session:
            result = await session.scalars(
                select(ExperimentMetric)
                .where(ExperimentMetric.paper_id == paper_id)
                .order_by(ExperimentMetric.experiment, ExperimentMetric.metric_name)
            )
            return list(result)
