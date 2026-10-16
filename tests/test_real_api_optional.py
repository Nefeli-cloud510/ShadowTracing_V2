import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import BaseModel, Field

from core.llm_gateway import LLMGateway
from core.rag_service import RAGService
from core.runtime_config import (
    get_bailian_knowledge_agent_id,
    get_bailian_workspace_id,
    get_dasyscope_api_key,
    get_knowledge_search_endpoint,
    real_api_tests_enabled,
)
from core.state_repository import UnifiedStateRepository
from scripts.run_real_api_two_round_demo import (
    ObservedLLMGateway,
    build_real_llm_control,
)


class SimpleStructuredResponse(BaseModel):
    answer: str = Field(min_length=1)


class FakeUsage:
    prompt_tokens = 7
    completion_tokens = 3


class FakeMessage:
    content = '{"answer":"ok"}'


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()


class FakeCompletions:
    def __init__(self, client: "FakeClient") -> None:
        self.client = client

    def create(self, **kwargs):
        self.client.calls += 1
        if self.client.calls <= self.client.fail_times:
            raise RuntimeError("boom")
        return FakeResponse()


class FakeChat:
    def __init__(self, client: "FakeClient") -> None:
        self.completions = FakeCompletions(client)


class FakeClient:
    def __init__(self, *, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.chat = FakeChat(self)


class ObservedGatewayWebAssemblyTest(unittest.TestCase):
    """Do not require an API key: verify the exact web assembly path."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.run_root = self.project_root / "run"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.repository = UnifiedStateRepository(self.project_root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_all_web_role_gateways_accept_role_and_use_base_path(self) -> None:
        bundle = build_real_llm_control(
            self.repository,
            self.run_root,
            model="test-model",
        )
        for role_name, gateway in bundle["gateways"].items():
            client = FakeClient()
            gateway.client = client

            def fixer(payload):
                payload["answer"] = "fixed"
                return payload

            response = gateway.generate_structured(
                system_prompt="sys",
                user_prompt=f"user-{role_name}-explicit-role",
                response_model=SimpleStructuredResponse,
                payload_fixer=fixer,
                role="audit",
            )
            self.assertEqual(response.answer, "fixed")
            self.assertTrue(gateway.call_history)
            self.assertEqual(gateway.call_history[-1]["role"], "audit")
            self.assertTrue(gateway.call_history[-1]["ok"])
            self.assertFalse(gateway.call_history[-1]["used_fallback"])

            gateway.client = FakeClient()
            gateway.generate_structured(
                system_prompt="sys",
                user_prompt=f"user-{role_name}-default-role",
                response_model=SimpleStructuredResponse,
                role=None,
            )
            self.assertEqual(gateway.call_history[-1]["role"], role_name)

    def test_observed_gateway_uses_base_cache(self) -> None:
        client = FakeClient()
        gateway = ObservedLLMGateway(
            role_name="audit_role",
            model="test-model",
            allow_fallback=False,
            use_cache=True,
            client=client,
        )
        user_prompt = f"cache-{id(self)}-shared"
        for _ in range(2):
            gateway.generate_structured(
                system_prompt="sys",
                user_prompt=user_prompt,
                response_model=SimpleStructuredResponse,
            )
        self.assertEqual(client.calls, 1)
        self.assertTrue(gateway.call_history[1]["cache_hit"])
        self.assertEqual(gateway.telemetry()["cache_hits"], 1)

    def test_observed_gateway_reports_retry_and_tokens(self) -> None:
        client = FakeClient(fail_times=1)
        gateway = ObservedLLMGateway(
            role_name="audit_role",
            model="test-model",
            max_retries=1,
            client=client,
        )
        with mock.patch("core.llm_gateway.time.sleep"):
            gateway.generate_structured(
                system_prompt="sys",
                user_prompt=f"retry-{id(self)}",
                response_model=SimpleStructuredResponse,
            )
        self.assertEqual(client.calls, 2)
        self.assertEqual(gateway.call_history[-1]["retries"], 1)
        self.assertEqual(gateway.telemetry()["retry_count"], 1)
        self.assertEqual(gateway.telemetry()["input_tokens"], 7)
        self.assertEqual(gateway.telemetry()["output_tokens"], 3)
        self.assertIn("elapsed_seconds", gateway.telemetry())

    def test_observed_gateway_never_silently_falls_back(self) -> None:
        client = FakeClient(fail_times=3)
        gateway = ObservedLLMGateway(
            role_name="audit_role",
            model="test-model",
            allow_fallback=False,
            max_retries=1,
            client=client,
        )
        fallback_calls = 0

        def fallback_factory():
            nonlocal fallback_calls
            fallback_calls += 1
            return {"answer": "fallback"}

        with mock.patch("core.llm_gateway.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                gateway.generate_structured(
                    system_prompt="sys",
                    user_prompt=f"no-fallback-{id(self)}",
                    response_model=SimpleStructuredResponse,
                    fallback_factory=fallback_factory,
                )
        self.assertEqual(fallback_calls, 0)
        self.assertFalse(gateway.call_history[-1]["used_fallback"])
        self.assertEqual(gateway.telemetry()["fallback_calls"], 0)
        self.assertIsNotNone(gateway.call_history[-1]["error"])

    def test_observed_gateway_fallback_records_single_entry(self) -> None:
        client = FakeClient(fail_times=3)
        gateway = ObservedLLMGateway(
            role_name="audit_role",
            model="test-model",
            allow_fallback=True,
            max_retries=1,
            client=client,
        )
        fallback_calls = 0

        def fallback_factory():
            nonlocal fallback_calls
            fallback_calls += 1
            return {"answer": "fallback"}

        with mock.patch("core.llm_gateway.time.sleep"):
            response = gateway.generate_structured(
                system_prompt="sys",
                user_prompt=f"allowed-fallback-{id(self)}",
                response_model=SimpleStructuredResponse,
                fallback_factory=fallback_factory,
            )
        self.assertEqual(response.answer, "fallback")
        self.assertEqual(fallback_calls, 1)
        self.assertEqual(len(gateway.call_history), 1)
        entry = gateway.call_history[0]
        self.assertFalse(entry["ok"])
        self.assertTrue(entry["used_fallback"])
        self.assertEqual(entry["error"], "llm_call_failed")
        telemetry = gateway.telemetry()
        self.assertEqual(telemetry["calls"], 1)
        self.assertEqual(telemetry["fallback_calls"], 1)
        self.assertEqual(telemetry["remote_success_calls"], 0)


@unittest.skipUnless(real_api_tests_enabled(), "未启用真实 API 可选测试")
class RealAPIOptionalIntegrationTest(unittest.TestCase):
    def test_real_bailian_model_structured_output(self) -> None:
        if not get_dasyscope_api_key():
            self.skipTest("未配置 DASHSCOPE_API_KEY / BAILIAN_API_KEY")

        gateway = LLMGateway()
        self.assertTrue(gateway.is_available())
        response = gateway.generate_structured(
            system_prompt="你是一个只输出 JSON 的助手。",
            user_prompt='请严格输出 {"answer":"ok"}',
            response_model=SimpleStructuredResponse,
        )
        self.assertTrue(response.answer)

    def test_real_knowledge_base_search_if_configured(self) -> None:
        if not get_dasyscope_api_key():
            self.skipTest("未配置 DASHSCOPE_API_KEY / BAILIAN_API_KEY")
        if not get_bailian_knowledge_agent_id():
            self.skipTest("未配置 BAILIAN_KNOWLEDGE_AGENT_ID")
        if not (get_bailian_workspace_id() or get_knowledge_search_endpoint()):
            self.skipTest("未配置 BAILIAN_WORKSPACE_ID 或 BAILIAN_KNOWLEDGE_SEARCH_ENDPOINT")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            rag_service = RAGService(project_root=project_root)
            results = rag_service.search_bailian_knowledge_base("DeltaDec Vsw By", max_results=2)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
