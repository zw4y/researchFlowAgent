from app.evaluation.generate_cases import normalize_answer_type


def test_unknown_generation_types_are_normalized() -> None:
    assert normalize_answer_type("mechanism") == "factual"
    assert normalize_answer_type("experimental setup") == "training"
    assert normalize_answer_type("efficiency") == "comparison"
    assert normalize_answer_type("unexpected-category") == "factual"
