"""Eligibility decisions preserve uncertainty and evaluate alternative rules."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("mcp_client", types.SimpleNamespace(load_mcp_tools=lambda: []))
import agent

EVIDENCE = "Doanh thu tối thiểu 10 tỷ. Công ty có doanh thu 12 tỷ. Phải đạt tiêu chí rủi ro. Có điểm rủi ro 80."


def condition(requirement, status, fact=""):
    return agent.EligibilityCondition(description="condition", requirement_quote=requirement, status=status, fact_quote=fact)


def analysis(*groups, complete=True, supported=True):
    return agent.EligibilityAnalysis(is_eligibility=True, complete_rules=complete, logic_supported=supported,
        alternatives=[agent.EligibilityGroup(conditions=list(group)) for group in groups])


class EligibilityTests(unittest.TestCase):
    def test_and_unknown_missing_risk(self):
        rule = analysis([condition("Doanh thu tối thiểu 10 tỷ.", "met", "Công ty có doanh thu 12 tỷ."),
                         condition("Phải đạt tiêu chí rủi ro.", "unknown")])
        verdict, groups = agent._eligibility_verdict(rule, EVIDENCE)
        self.assertEqual(verdict, "unknown")
        self.assertTrue(agent._eligibility_response(verdict, groups, "Có đủ điều kiện không?").startswith("Chưa đủ thông tin"))

    def test_and_success_and_failure(self):
        first = condition("Doanh thu tối thiểu 10 tỷ.", "met", "Công ty có doanh thu 12 tỷ.")
        self.assertEqual(agent._eligibility_verdict(analysis([first]), EVIDENCE)[0], "eligible")
        second = condition("Phải đạt tiêu chí rủi ro.", "not_met", "Có điểm rủi ro 80.")
        self.assertEqual(agent._eligibility_verdict(analysis([first, second]), EVIDENCE)[0], "ineligible")

    def test_or_requires_every_alternative_to_fail(self):
        no = condition("Doanh thu tối thiểu 10 tỷ.", "not_met", "Công ty có doanh thu 12 tỷ.")
        unknown = condition("Phải đạt tiêu chí rủi ro.", "unknown")
        yes = condition("Phải đạt tiêu chí rủi ro.", "met", "Có điểm rủi ro 80.")
        self.assertEqual(agent._eligibility_verdict(analysis([no], [unknown]), EVIDENCE)[0], "unknown")
        self.assertEqual(agent._eligibility_verdict(analysis([no], [yes]), EVIDENCE)[0], "eligible")
        self.assertEqual(agent._eligibility_verdict(analysis([no], [no]), EVIDENCE)[0], "ineligible")

    def test_percentage_requires_both_valid_quotes_and_recent_year(self):
        from datetime import date
        year = date.today().year - 1
        rule = "Khoản vay đề nghị không vượt quá 30% doanh thu năm gần nhất"
        loan = "Công ty A đề nghị khoản vay 2 tỷ đồng."
        revenue = f"Doanh thu năm {year} của Công ty A là 12 tỷ đồng."
        evidence = " ".join([rule, loan, revenue])
        item = agent.EligibilityCondition(description="ratio", requirement_quote=rule, status="unknown", fact_quotes=[loan, revenue])
        status, calculation = agent._verify_percentage_condition(item, evidence, "test")
        self.assertEqual(status, "met")
        self.assertIn("3.6", calculation)
        self.assertEqual(agent._eligibility_verdict(analysis([item]), evidence)[0], "eligible")
        missing = item.model_copy(update={"fact_quotes": [loan]})
        self.assertEqual(agent._verify_percentage_condition(missing, evidence)[0], "unknown")
        self.assertEqual(agent._verify_percentage_condition(item, rule + loan)[0], "unknown")
        old = revenue.replace(str(year), str(year - 1))
        old_item = item.model_copy(update={"fact_quotes": [loan, old]})
        self.assertEqual(agent._verify_percentage_condition(old_item, rule + loan + old)[0], "unknown")

    def test_unsupported_quotes_and_incomplete_rules_are_uncertain(self):
        invented = condition("Doanh thu tối thiểu 10 tỷ.", "met", "Công ty có doanh thu 99 tỷ.")
        self.assertEqual(agent._eligibility_verdict(analysis([invented]), EVIDENCE)[0], "unknown")
        good = condition("Doanh thu tối thiểu 10 tỷ.", "met", "Công ty có doanh thu 12 tỷ.")
        self.assertEqual(agent._eligibility_verdict(analysis([good], complete=False), EVIDENCE)[0], "unknown")
        self.assertEqual(agent._eligibility_verdict(analysis([good], supported=False), EVIDENCE)[0], "unknown")


class EligibilityFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_does_not_approve_explicit_eligibility(self):
        with patch.object(agent, "_providers", return_value=[]):
            text = await agent._assess_eligibility("Công ty có đủ điều kiện không?", EVIDENCE, "test")
        self.assertIn("dịch vụ model không khả dụng", text)

    async def test_final_answer_does_not_claim_eligibility_when_risk_unknown(self):
        from langchain_core.messages import AIMessage, HumanMessage
        class Model:
            def with_structured_output(self, *args, **kwargs): return self
            async def ainvoke(self, messages, config=None):
                return analysis([condition("Doanh thu tối thiểu 10 tỷ.", "met", "Công ty có doanh thu 12 tỷ."),
                                 condition("Phải đạt tiêu chí rủi ro.", "unknown")])
        state = {"messages": [HumanMessage(content="Công ty có đủ điều kiện không?"),
                    AIMessage(content="Document retrieval result:\n" + EVIDENCE,
                              additional_kwargs={"rag_run_id": "test"})]}
        with patch.object(agent, "_providers", return_value=[("fake", lambda **kwargs: Model())]):
            result = await agent._document_answer(state)
        self.assertTrue(result.content.startswith("Chưa đủ thông tin"))
        self.assertIn("Chưa xác định", result.content)

    def test_retrieval_notes_are_not_evidence(self):
        evidence = "Document retrieval result:\nMissing: invented fact\n\nDocument 1\nSource: a.md\nMetadata: \nContent: actual fact"
        self.assertFalse(agent._evidence_contains(evidence, "invented fact"))
        self.assertTrue(agent._evidence_contains(evidence, "actual fact"))

    async def test_independent_review_adds_missing_risk(self):
        initial = analysis([condition("Doanh thu tối thiểu 10 tỷ.", "met", "Công ty có doanh thu 12 tỷ.")])
        review = agent.EligibilityReview(coverage_confirmed=False, missing=[agent.MissingRequirement(
            requirement_quote="Phải đạt tiêu chí rủi ro.", group_index=0)])
        class Model:
            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                return self
            async def ainvoke(self, messages, config=None):
                return review if self.schema is agent.EligibilityReview else initial
        with patch.object(agent, "_providers", return_value=[("fake", lambda **kwargs: Model())]):
            result = await agent._assess_eligibility("Có đủ điều kiện không?", EVIDENCE, "test")
        self.assertTrue(result.startswith("Chưa đủ thông tin"))
        self.assertIn("Phải đạt tiêu chí rủi ro.", result)
        self.assertIn("Chưa xác định", result)

    async def test_loan_fragment_recovers_facts_and_surfaces_missing_risk(self):
        from datetime import date as real_date
        class Clock(real_date):
            @classmethod
            def today(cls): return cls(2026, 11, 17)
        sample = Path(__file__).resolve().parents[1] / "sample-data"
        policy = (sample / "loan-credit-policy.md").read_text().strip()
        company = (sample / "loan-company-a.md").read_text().strip()
        evidence = ("Document retrieval result:\nDocument 1\nSource: loan-credit-policy.md\nMetadata: \nContent: " + policy
                    + "\n\nDocument 2\nSource: loan-company-a.md\nMetadata: \nContent: " + company)
        ratio = agent.EligibilityCondition(description="ratio", requirement_quote="không vượt quá 30% doanh thu năm gần nhất",
            status="unknown", fact_quotes=["Doanh thu năm 2025 của Công ty A là 12 tỷ đồng."])
        initial = analysis([ratio], complete=False, supported=False)
        class Model:
            def with_structured_output(self, schema, **kwargs): self.schema = schema; return self
            async def ainvoke(self, messages, config=None):
                return agent.EligibilityReview(coverage_confirmed=False) if self.schema is agent.EligibilityReview else initial
        with patch.object(agent, "_providers", return_value=[("fake", lambda **kwargs: Model())]), patch.object(agent, "date", Clock):
            result = await agent._assess_eligibility("Công ty A có đủ điều kiện vay không?", evidence, "test")
        self.assertTrue(result.startswith("Chưa đủ thông tin"))
        self.assertIn("Đạt — Điều kiện: “không vượt quá 30%", result)
        self.assertIn("Công ty A đề nghị khoản vay 2 tỷ đồng.", result)
        self.assertIn("30% × 12 = 3.6", result)
        self.assertIn("doanh nghiệp phải đạt các tiêu chí trong chính sách đánh giá rủi ro", result)

    async def test_ratio_does_not_overwrite_other_conditions_in_same_sentence(self):
        from datetime import date as real_date
        class Clock(real_date):
            @classmethod
            def today(cls): return cls(2026, 11, 17)
        sample = Path(__file__).resolve().parents[1] / "sample-data"
        policy = (sample / "loan-credit-policy.md").read_text().strip()
        company = (sample / "loan-company-a.md").read_text().strip()
        evidence = ("Document retrieval result:\nDocument 1\nSource: policy.md\nMetadata: \nContent: " + policy
                    + "\n\nDocument 2\nSource: company.md\nMetadata: \nContent: " + company)
        initial = analysis([condition("doanh thu năm gần nhất từ 10 tỷ đồng trở lên", "not_met", "Doanh thu năm 2025 của Công ty A là 12 tỷ đồng."),
                            condition("hoạt động ít nhất 3 năm", "unknown"),
                            condition("khoản vay đề nghị không vượt quá 30% doanh thu năm gần nhất", "unknown")], complete=False)
        class Model:
            def with_structured_output(self, schema, **kwargs): self.schema = schema; return self
            async def ainvoke(self, messages, config=None):
                return agent.EligibilityReview(coverage_confirmed=False) if self.schema is agent.EligibilityReview else initial
        with patch.object(agent, "_providers", return_value=[("fake", lambda **kwargs: Model())]), patch.object(agent, "date", Clock):
            result = await agent._assess_eligibility("Công ty A có đủ điều kiện vay không?", evidence, "test")
        lines = [line for line in result.splitlines() if line.startswith("- ")]
        self.assertIn("Không đạt — Điều kiện: “doanh thu", lines[0])
        self.assertIn("Chưa xác định — Điều kiện: “hoạt động", lines[1])
        self.assertNotIn("30% ×", lines[0])
        self.assertNotIn("30% ×", lines[1])
        self.assertNotIn("khoản vay 2 tỷ", lines[0])
        self.assertNotIn("khoản vay 2 tỷ", lines[1])
        self.assertIn("Đạt — Điều kiện: “khoản vay", lines[2])
        self.assertIn("30% × 12 = 3.6", lines[2])

    async def test_ambiguous_loan_amount_does_not_calculate(self):
        policy = "Khoản vay đề nghị không vượt quá 30% doanh thu năm gần nhất."
        evidence = policy + " Công ty A đề nghị khoản vay 2 tỷ đồng. Công ty A đề nghị khoản vay 4 tỷ đồng. Doanh thu năm 2025 của Công ty A là 12 tỷ đồng."
        item = condition("không vượt quá 30% doanh thu năm gần nhất", "unknown")
        rules = analysis([item])
        agent._prepare_conditions(rules, evidence, "Công ty A có đủ điều kiện vay không?", "test")
        self.assertFalse(item.fact_quotes)
        self.assertEqual(agent._verify_percentage_condition(item, evidence)[0], "unknown")

    async def test_timeout_tries_next_provider_with_gemini_json_schema(self):
        import asyncio
        used = []
        class Model:
            def __init__(self, provider): self.provider = provider
            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                used.append((self.provider, schema, kwargs.get("method")))
                return self
            async def ainvoke(self, messages, config=None):
                if self.provider == "OpenAI":
                    await asyncio.sleep(0.05)
                if self.schema is agent.EligibilityReview:
                    return agent.EligibilityReview(coverage_confirmed=True)
                return analysis([condition("Doanh thu tối thiểu 10 tỷ.", "met", "Công ty có doanh thu 12 tỷ.")])
        providers = [(name, lambda name=name, **kwargs: Model(name)) for name in ("OpenAI", "Gemini")]
        with patch.object(agent, "_providers", return_value=providers), patch.object(agent, "ELIGIBILITY_TIMEOUT_SECONDS", 0.01):
            result = await agent._assess_eligibility("Có đủ điều kiện không?", "Doanh thu tối thiểu 10 tỷ. Công ty có doanh thu 12 tỷ.", "test")
        self.assertTrue(result.startswith("Đủ điều kiện"))
        self.assertIn(("Gemini", agent.EligibilityAnalysis, "json_schema"), used)
        self.assertIn(("Gemini", agent.EligibilityReview, "json_schema"), used)

    async def test_ordinary_question_keeps_normal_path(self):
        class Model:
            def with_structured_output(self, *args, **kwargs): return self
            async def ainvoke(self, messages, config=None): return agent.EligibilityAnalysis(is_eligibility=False)
        with patch.object(agent, "_providers", return_value=[("fake", lambda **kwargs: Model())]):
            self.assertIsNone(await agent._assess_eligibility("Doanh thu là bao nhiêu?", EVIDENCE, "test"))


if __name__ == "__main__":
    unittest.main()
