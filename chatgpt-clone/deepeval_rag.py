"""Evaluate real document answers against a fixed, local evidence fixture.

This isolates answer generation; it does not evaluate retrieval quality.
"""
import asyncio
import os
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_core.messages import AIMessage, HumanMessage

import agent


class AgentJudge(DeepEvalBaseLLM):
    """Use the same configured provider fallback as the application."""

    def __init__(self):
        super().__init__(model="configured-agent-providers")

    def load_model(self):
        return self

    def get_model_name(self):
        return "configured-agent-providers"

    def generate(self, prompt, schema=None):
        return asyncio.run(self.a_generate(prompt, schema))

    async def a_generate(self, prompt, schema=None):
        errors = []
        for name, factory in agent._providers():
            try:
                model = factory(streaming=False)
                if schema:
                    model = model.with_structured_output(schema, method="function_calling")
                result = await agent._invoke_with_timeout(model, [HumanMessage(content=prompt)])
                return result if schema else str(result.content)
            except Exception as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        raise RuntimeError("No judge provider succeeded: " + "; ".join(errors))


async def main():
    if not agent._providers():
        raise RuntimeError("Configure an OpenAI, Gemini, or Groq API key for the application and judge.")
    path = Path(__file__).parent / "sample-data" / "loan-company-a.md"
    context = path.read_text(encoding="utf-8").strip()
    evidence = f"Document retrieval result:\nDocument 1\nSource: {path.name}\nMetadata: \nContent: {context}"
    questions = ["Doanh thu năm 2025 của Công ty A là bao nhiêu?", "Công ty A đề nghị khoản vay bao nhiêu?"]
    threshold = float(os.getenv("DEEPEVAL_RAG_THRESHOLD", "0.7"))
    if not 0 <= threshold <= 1:
        raise ValueError("DEEPEVAL_RAG_THRESHOLD must be between 0 and 1")
    failed = False
    for question in questions:
        state = {"messages": [HumanMessage(content=question), AIMessage(content=evidence)]}
        answer = (await agent._document_answer(state)).content
        print(f"\nQuestion: {question}\nAnswer: {answer}", flush=True)
        case = LLMTestCase(input=question, actual_output=answer, retrieval_context=[context])
        for metric_type in (FaithfulnessMetric, AnswerRelevancyMetric):
            metric = metric_type(model=AgentJudge(), threshold=threshold, include_reason=True)
            try:
                await metric.a_measure(case)
                passed = metric.is_successful()
                print(f"{metric_type.__name__}: {'PASS' if passed else 'FAIL'} score={metric.score} reason={metric.reason}", flush=True)
                failed |= not passed
            except Exception as error:
                failed = True
                print(f"{metric_type.__name__}: ERROR {type(error).__name__}: {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
