"""Check selection, evidence preservation, and multi-query activation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.documents import Document
from rag import prepare_multi_query_context


def doc(text, source, **metadata):
    return Document(page_content=text, metadata={"source": source, **metadata})


class ContextPreparationTests(unittest.TestCase):
    def test_filters_irrelevant_content_and_contextualizes_only_explicit_metadata(self):
        documents = [
            doc("Doanh thu 12 tỷ đồng.\nDoanh thu 12 tỷ đồng.", "company.md", department="HR"),
            doc("Chính sách tín dụng yêu cầu đánh giá rủi ro.", "policy.md"),
            doc("Rainfall prediction with machine learning.", "rain.pdf"),
        ]
        result = prepare_multi_query_context(documents, "doanh thu tín dụng rủi ro", ["doanh thu", "tín dụng rủi ro"], "test")
        self.assertEqual({item.metadata['source'] for item in result}, {"company.md", "policy.md"})
        company = next(item for item in result if item.metadata['source'] == 'company.md')
        policy = next(item for item in result if item.metadata['source'] == 'policy.md')
        self.assertIn("Document metadata (as provided): department: HR", company.page_content)
        self.assertEqual(company.page_content.count("Doanh thu 12 tỷ đồng."), 1)
        self.assertEqual(policy.page_content, documents[1].page_content)
        self.assertEqual(documents[0].page_content.count("Doanh thu 12 tỷ đồng."), 2)

    def test_orders_most_relevant_at_start_and_second_at_end_with_cap(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
        documents = [doc(' '.join(words[:n]), str(n)) for n in range(7, 0, -1)]
        result = prepare_multi_query_context(documents, ' '.join(words), ["alpha", "beta"], "test")
        self.assertEqual(len(result), 6)
        self.assertEqual([item.metadata['source'] for item in result], ['7', '5', '3', '2', '4', '6'])

    def test_loan_sample_excludes_unrelated_hr_and_rainfall(self):
        sample = Path(__file__).resolve().parents[1] / "sample-data"
        names = ["loan-company-a.md", "loan-credit-policy.md", "loan-risk-policy.md", "hr-leave-policy.md"]
        documents = [doc((sample / name).read_text(), name) for name in names]
        documents.append(doc("Rainfall prediction using machine learning.", "rain.pdf"))
        result = prepare_multi_query_context(documents,
            "Công ty A có đủ điều kiện vay 2 tỷ đồng không? Phân tích doanh thu, chính sách tín dụng và rủi ro.",
            ["Doanh thu Công ty A?", "Điều kiện tín dụng?", "Tiêu chí rủi ro?"], "test")
        self.assertEqual({item.metadata['source'] for item in result}, set(names[:3]))

    def test_skips_single_query_or_single_source(self):
        documents = [doc('one', 'same'), doc('two', 'same')]
        self.assertIs(prepare_multi_query_context(documents, 'one', ['one', 'two'], 'test'), documents)
        different_sources = [doc('one', 'a'), doc('two', 'b')]
        self.assertIs(prepare_multi_query_context(different_sources, 'one', ['one'], 'test'), different_sources)

    def test_no_lexical_matches_preserves_evidence_instead_of_dropping_all(self):
        documents = [doc('evidence one', 'a'), doc('evidence two', 'b')]
        result = prepare_multi_query_context(documents, 'unmatched', ['unmatched', 'absent'], 'test')
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
