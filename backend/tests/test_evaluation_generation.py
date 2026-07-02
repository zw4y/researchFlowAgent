from app.db.models import Chunk, Paper
from app.evaluation.generate_cases import (
    GeneratedBatch,
    GeneratedCase,
    build_page_context,
    convert_cases,
    label_status,
)


def test_build_page_context_preserves_page_and_chunk_order() -> None:
    chunks = [
        Chunk(
            paper_id="paper-1",
            page=2,
            chunk_index=1,
            text="second",
            token_count=1,
            vector_id="v2",
            index_profile="profile",
        ),
        Chunk(
            paper_id="paper-1",
            page=2,
            chunk_index=0,
            text="first",
            token_count=1,
            vector_id="v1",
            index_profile="profile",
        ),
    ]

    pages = build_page_context(chunks, max_chars_per_page=100)

    assert pages == {2: "first\nsecond"}


def test_numeric_label_is_auto_checked_only_when_values_exist_on_page() -> None:
    grounded = GeneratedCase(
        question="What score is reported?",
        expected_answer="The score is 91.5%.",
        pages=[3],
        answer_type="numeric_table",
    )
    unsupported = grounded.model_copy(update={"expected_answer": "The score is 99.9%."})

    assert label_status(grounded, {3: "Accuracy reaches 91.5%."}) == "auto_checked"
    assert label_status(unsupported, {3: "Accuracy reaches 91.5%."}) == "machine_generated"


def test_convert_cases_creates_stable_dev_test_split() -> None:
    paper = Paper(
        id="12345678-1234-1234-1234-123456789012",
        title="Evaluation Paper",
        original_filename="paper.pdf",
        stored_filename="paper.pdf",
        sha256="a" * 64,
        page_count=3,
        status="ready",
        index_status="ready",
    )
    generated = GeneratedBatch(
        cases=[
            GeneratedCase(
                question=f"What result is reported in experiment {index}?",
                expected_answer=f"Experiment {index} reports {index}.0.",
                pages=[1],
                answer_type="numeric_table",
            )
            for index in range(1, 9)
        ]
    )

    cases = convert_cases(
        paper,
        generated,
        {1: "Experiments 1 2 3 4 5 6 7 8 report 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0."},
        questions_per_paper=8,
    )

    assert len(cases) == 8
    assert [case.split for case in cases].count("dev") == 2
    assert [case.split for case in cases].count("test") == 6
    assert all(case.relevant_pages[0].paper_title == "Evaluation Paper" for case in cases)
