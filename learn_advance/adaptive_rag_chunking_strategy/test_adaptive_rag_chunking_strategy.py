"""Unit tests for the dependency-free Adaptive Chunking Strategy Selector."""

import unittest

from adaptive_rag_chunking_strategy import DocumentReadError, read_document_bytes, select_chunking_strategy


class ChunkingStrategySelectorTests(unittest.TestCase):
    def assert_strategy(self, expected: str, **document: object) -> None:
        result = select_chunking_strategy(**document)
        self.assertEqual(expected, result["strategy"])
        self.assertTrue(result["reason"])
        self.assertIn("signals", result)
        self.assertIn("file_type", result["signals"])

    def test_markdown_headings_choose_structure_based(self) -> None:
        self.assert_strategy(
            "structure_based",
            file_name="architecture.md",
            content="# Overview\nText\n\n## Components\nText\n\n## Deployment\nText",
        )

    def test_source_definitions_choose_structure_based(self) -> None:
        self.assert_strategy(
            "structure_based",
            file_name="service.py",
            content="class Service:\n    pass\n\ndef start_server():\n    pass",
        )

    def test_parser_supplied_pdf_sections_choose_structure_based(self) -> None:
        self.assert_strategy(
            "structure_based",
            file_name="report.pdf",
            metadata={"section_boundaries": ["Introduction", "Results"]},
        )

    def test_ordinary_prose_chooses_recursive(self) -> None:
        self.assert_strategy(
            "recursive",
            file_type="text",
            content="This report explains the project goals. It introduces the context.\n\nThe second paragraph explains the implementation in more detail.",
        )

    def test_flat_text_chooses_fixed_size(self) -> None:
        self.assert_strategy("fixed_size", file_name="stream.txt", content="alpha beta gamma delta")

    def test_topic_shifts_choose_semantic(self) -> None:
        self.assert_strategy(
            "semantic",
            file_type="text",
            content=(
                "Topic: Volcanoes erupt when magma pressure fractures surrounding rock layers. Ash clouds affect aviation.\n\n"
                "Topic: Jazz improvisation uses harmony rhythm phrasing and ensemble interaction during performances.\n\n"
                "Topic: Database indexing improves query planning through btree statistics caching and storage layouts."
            ),
        )

    def test_irregular_interdependent_content_recommends_agentic(self) -> None:
        self.assert_strategy(
            "agentic",
            file_type="pdf",
            content="Fragmented layout with nested references.",
            metadata={"complex_layout": True, "cross_references": True},
        )

    def test_insufficient_input_chooses_recursive(self) -> None:
        self.assert_strategy("recursive")

    def test_plain_text_file_reader_returns_text_and_metadata(self) -> None:
        content, metadata = read_document_bytes(b"# Overview\nText\n\n## Details\nMore text", "notes.md")
        self.assertIn("Overview", content)
        self.assertEqual(2, metadata["heading_count"])
        self.assertTrue(metadata["has_structure"])

    def test_binary_unknown_file_is_rejected(self) -> None:
        with self.assertRaises(DocumentReadError):
            read_document_bytes(b"\xff\xfe\x00", "unknown.bin")

    def test_irregular_file_reader_infers_agentic_signals(self) -> None:
        content = (
            "Fragment A.\n"
            "    - level one\n"
            "        - level two\n"
            "            - level three\n"
            "See section 2.1 and refer to appendix A.\n"
            "See section 3.4 and refer to appendix B."
        )
        extracted, metadata = read_document_bytes(content.encode(), "fragments.txt")
        result = select_chunking_strategy(
            file_name="fragments.txt", content=extracted, metadata=metadata
        )
        self.assertEqual("agentic", result["strategy"])


if __name__ == "__main__":
    unittest.main()
