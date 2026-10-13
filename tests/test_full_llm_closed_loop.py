"""Closed-loop integration tests for the stubbed-agent pipeline.

This module intentionally does NOT claim to be a real LLM test. It drives
HumanControlService with deterministic stub agents, so it verifies control
flow, state transitions, protocol execution and trace structure, not model
quality. The three test layers live in:

- Deterministic execution: tests/test_calendar_samples.py,
  tests/test_unified_execution.py, tests/test_experiment_dispatcher.py,
  tests/test_robustness_diagnostics.py, tests/test_evaluation_plan.py
- Stubbed agent integration: this file
- Explicitly enabled real API: tests/test_real_api_optional.py
"""

import unittest

from core.central_controller_llm import CentralControllerLLM
from core.control_unified import HumanControlService
from core.planner_unified import PlannerOutputBuilder
from core.round_orchestrator import RoundOrchestrator
from core.scientific_interpreter_llm import ScientificInterpreterLLM
from tests.test_decision_layer import (
    StubCandidateExperimentDesigner,
    StubCandidateExperimentWriter,
)
from tests.test_llm_planner_integration import (
    StubExperimentPlanner,
    StubHypothesisProposer,
    StubLLMGateway,
    StubRAGService,
    StubScientificInterpreterGateway,
    StubScientificQuestioner,
)
from tests.test_system_audit_closure import SystemAuditClosureTest


class StubbedAgentIntegrationTest(SystemAuditClosureTest):
    def _build_control(self) -> HumanControlService:
        return HumanControlService(
            self.repository,
            project_root=self.project_root,
            run_shap=False,
            planner_mode="llm",
            planner_output_builder=PlannerOutputBuilder(
                mode="llm",
                central_controller=CentralControllerLLM(gateway=StubLLMGateway()),
            ),
            scientific_interpreter=ScientificInterpreterLLM(gateway=StubScientificInterpreterGateway()),
            rag_service=StubRAGService(),
            hypothesis_proposer=StubHypothesisProposer(),
            scientific_questioner=StubScientificQuestioner(),
            experiment_planner=StubExperimentPlanner(),
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )

    def test_two_round_closure_with_stubbed_agents(self) -> None:
        self.assertEqual(self._build_control().planner_mode, "llm")
        payload = self._run_two_rounds()

        self.assertEqual(payload["initial_review"]["status"], "awaiting_human_review")
        self.assertEqual(payload["round1"]["review"]["status"], "awaiting_round_decision")
        self.assertEqual(payload["round2"]["review"]["status"], "awaiting_round_decision")

        self.assertTrue(
            any(trace["stage"] == "scientific_interpreter" for trace in payload["round1"]["review"]["reasoning_traces"])
        )
        self.assertTrue(
            any(trace["stage"] == "scientific_interpreter" for trace in payload["round2"]["review"]["reasoning_traces"])
        )

        plan_after_round1 = payload["plan_after_round1"]
        plan_after_round2 = payload["plan_after_round2"]
        self.assertEqual(plan_after_round1["planner_output"].planner_mode, "llm")
        self.assertEqual(plan_after_round2["planner_output"].planner_mode, "llm")
        self.assertTrue(
            any("rag_project:" in note for note in plan_after_round1["planner_input"].planner_guidance)
        )
        self.assertTrue(
            any("questioner_gap:" in note for note in plan_after_round1["planner_input"].planner_guidance)
        )
        traces = plan_after_round1["planner_input"].recent_reasoning_traces
        referenced_variables = [
            variable
            for trace in traces
            for variable in trace.referenced_variables
        ]
        self.assertIn("Np", referenced_variables)
        self.assertTrue(any("Solar wind density" in trace.summary for trace in traces))

        self.assertTrue(
            any(note == "planner_refinement:实验规划者llm" for note in payload["round1"]["protocol"].notes)
        )
        self.assertTrue(
            any(step.action == "planner_hypothesis_focus" for step in payload["round1"]["protocol"].steps)
        )
        self.assertTrue(
            any(note == "planner_refinement:实验规划者llm" for note in payload["round2"]["protocol"].notes)
        )

        uncertainties = self.repository.load_uncertainties()
        self.assertTrue(
            any("Solar wind density 条件下成立" in record.question for record in uncertainties.records)
        )

        decision_log = self.repository.load_decision_log()
        decision_types = [entry.decision_type for entry in decision_log.decisions]
        self.assertGreaterEqual(decision_types.count("scientific_interpreter_applied"), 2)
        self.assertGreaterEqual(decision_types.count("planner_output_generated"), 2)
        self.assertGreaterEqual(decision_types.count("planner_output_applied"), 2)

        candidate_set = self.repository.load_candidate_experiments()
        self.assertTrue(candidate_set.candidates)
        self.assertTrue(
            any("llm_controller_selected" in candidate.design.notes for candidate in candidate_set.candidates)
        )


if __name__ == "__main__":
    unittest.main()
