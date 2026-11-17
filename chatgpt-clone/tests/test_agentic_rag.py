"""Exercise evidence assessment and bounded follow-up retrieval without model calls."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("mcp_client", types.SimpleNamespace(load_mcp_tools=lambda: []))

from langchain_core.documents import Document
import agent


class FakeModel:
    def __init__(self, responses):
        self.responses = responses

    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def document(text):
    return Document(page_content=text, metadata={"source": text})


class AgenticRagTests(unittest.IsolatedAsyncioTestCase):
    async def run_flow(self, responses, retrieve):
        responses = iter(responses)
        with patch.object(agent, "_providers", return_value=[("fake", lambda **kw: FakeModel(responses))]), \
             patch.object(agent, "retrieve_documents", side_effect=retrieve):
            return await agent._retrieve_question_evidence("original question", {"department": "HR"})

    async def test_missing_evidence_triggers_followup_and_merges(self):
        calls = []
        def retrieve(query, filters):
            calls.append((query, filters))
            return [document("first")] if query == "initial" else [document("first"), document("second")]
        result = await self.run_flow([
            agent.RetrievalPlan(questions=["initial"]),
            agent.EvidenceAssessment(sufficient=False, missing=["second fact"], follow_up_questions=["followup"]),
            agent.EvidenceAssessment(sufficient=True),
        ], retrieve)
        self.assertEqual(calls, [("initial", {"department": "HR"}), ("followup", {"department": "HR"})])
        self.assertEqual(result.count("Content: first"), 1)
        self.assertIn("Content: second", result)

    async def test_no_new_evidence_stops_and_reports_missing(self):
        calls = []
        def retrieve(query, filters):
            calls.append(query)
            return [document("same")]
        result = await self.run_flow([
            agent.RetrievalPlan(questions=["initial"]),
            agent.EvidenceAssessment(sufficient=False, missing=["risk"], follow_up_questions=["risk query"]),
            agent.EvidenceAssessment(sufficient=False, missing=["risk"], follow_up_questions=["another query"]),
        ], retrieve)
        self.assertEqual(calls, ["initial", "risk query"])
        self.assertIn("Potentially missing: risk", result)

    async def test_assessment_failure_reports_uncertainty(self):
        result = await self.run_flow([
            agent.RetrievalPlan(questions=["initial"]), RuntimeError("model failed")
        ], lambda query, filters: [document("evidence")])
        self.assertIn("Evidence coverage could not be assessed", result)

    async def test_round_limit_and_failed_retrieval(self):
        calls = []
        def retrieve(query, filters):
            calls.append(query)
            if query == "failed":
                raise RuntimeError("failed")
            return [document(query)]
        result = await self.run_flow([
            agent.RetrievalPlan(questions=["initial", "failed"]),
            agent.EvidenceAssessment(sufficient=False, missing=["one"], follow_up_questions=["first followup"]),
            agent.EvidenceAssessment(sufficient=False, missing=["two"], follow_up_questions=["second followup"]),
            agent.EvidenceAssessment(sufficient=False, missing=["three"], follow_up_questions=["extra"]),
        ], retrieve)
        self.assertEqual(calls, ["initial", "failed", "first followup", "second followup"])
        self.assertIn("Some retrievals failed", result)
        self.assertIn("Potentially missing: three", result)


if __name__ == "__main__":
    unittest.main()
