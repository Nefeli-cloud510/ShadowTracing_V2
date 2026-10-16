import unittest

from core.hypothesis_state_machine import resolve_status
from core.unified_schema import SupportHistoryEntry


def _experimental_entry(
    round_id: int,
    score: float,
    *,
    classification: str,
    direction_matched: bool = True,
    event: str = "evaluation_writeback:E",
) -> SupportHistoryEntry:
    return SupportHistoryEntry(
        round=round_id,
        score=score,
        event=event,
        evidence_source="experimental",
        evidence_classification=classification,
        direction_matched=direction_matched,
    )


class ConvergenceRequiresExperimentalEvidenceTest(unittest.TestCase):
    def test_two_stable_improvement_rounds_converge(self) -> None:
        history = [
            _experimental_entry(1, 0.75, classification="stable_improvement"),
        ]
        status = resolve_status(
            previous_status="active",
            support_after=0.80,
            support_history=history,
            current_round=2,
            evidence_source="experimental",
            evidence_classification="stable_improvement",
            direction_matched=True,
        )
        self.assertEqual(status, "converged")

    def test_questioning_history_alone_cannot_converge(self) -> None:
        history = [
            SupportHistoryEntry(round=1, score=0.75, event="scientific_questioning"),
            SupportHistoryEntry(round=2, score=0.80, event="scientific_questioning"),
        ]
        status = resolve_status(
            previous_status="active",
            support_after=0.82,
            support_history=history,
            current_round=3,
            evidence_source="experimental",
            evidence_classification="stable_improvement",
            direction_matched=True,
        )
        self.assertEqual(status, "active")

    def test_inconclusive_last_experiment_does_not_converge(self) -> None:
        history = [
            _experimental_entry(1, 0.75, classification="stable_improvement"),
        ]
        status = resolve_status(
            previous_status="active",
            support_after=0.80,
            support_history=history,
            current_round=2,
            evidence_source="experimental",
            evidence_classification="inconclusive",
            direction_matched="partial",
        )
        self.assertEqual(status, "active")

    def test_no_stable_gain_experiment_does_not_converge(self) -> None:
        history = [
            _experimental_entry(1, 0.75, classification="stable_improvement"),
        ]
        status = resolve_status(
            previous_status="active",
            support_after=0.78,
            support_history=history,
            current_round=2,
            evidence_source="experimental",
            evidence_classification="no_stable_gain",
            direction_matched="partial",
        )
        self.assertEqual(status, "active")

    def test_wrong_direction_stable_improvement_does_not_converge(self) -> None:
        history = [
            _experimental_entry(
                1,
                0.75,
                classification="stable_improvement",
                direction_matched=False,
            ),
        ]
        status = resolve_status(
            previous_status="active",
            support_after=0.75,
            support_history=history,
            current_round=2,
            evidence_source="experimental",
            evidence_classification="stable_improvement",
            direction_matched=False,
        )
        self.assertEqual(status, "active")

    def test_stable_degradation_only_converges_when_direction_matched(self) -> None:
        history = [
            _experimental_entry(
                1,
                0.75,
                classification="stable_degradation",
                direction_matched=True,
            ),
        ]
        matched = resolve_status(
            previous_status="active",
            support_after=0.76,
            support_history=history,
            current_round=2,
            evidence_source="experimental",
            evidence_classification="stable_degradation",
            direction_matched=True,
        )
        self.assertEqual(matched, "converged")

        not_matched = resolve_status(
            previous_status="active",
            support_after=0.76,
            support_history=[
                _experimental_entry(
                    1,
                    0.75,
                    classification="stable_degradation",
                    direction_matched=False,
                )
            ],
            current_round=2,
            evidence_source="experimental",
            evidence_classification="stable_degradation",
            direction_matched=False,
        )
        self.assertEqual(not_matched, "active")


if __name__ == "__main__":
    unittest.main()
