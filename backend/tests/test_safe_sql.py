import pytest
from app.core.errors import AppError
from app.services.metrics import SafeSQLValidator


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM experiment_metrics",
        "SELECT paper_id FROM papers",
        "SELECT * FROM experiment_metrics",
        "SELECT paper_id FROM experiment_metrics; DROP TABLE papers",
        "SELECT pg_sleep(10) FROM experiment_metrics",
    ],
)
def test_safe_sql_rejects_dangerous_queries(sql):
    with pytest.raises(AppError):
        SafeSQLValidator().validate(sql)


def test_safe_sql_adds_limit():
    validated = SafeSQLValidator().validate(
        "SELECT paper_id, metric_name, metric_value FROM experiment_metrics"
    )
    assert "LIMIT 50" in validated
