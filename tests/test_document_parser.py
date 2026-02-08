"""Unit tests for DocumentParser and ParsedDocument.

Covers:
- ParsedDocument dataclass: defaults, has_images(), get_full_content()
- DocumentParser.parse(): routing by extension, unsupported types, missing files
- _parse_pdf(): PyMuPDF path, pdfplumber fallback, PyPDF2 fallback, no-lib error
- _parse_docx(): paragraph/table extraction, image extraction, missing lib
- _parse_image(): all MIME types, base64 encoding, text placeholder
- _parse_text(): UTF-8 reading
- Edge cases: empty documents, blank pages, image extraction failures
"""

import base64
import os
import types
import unittest
from unittest.mock import MagicMock, patch

from datarecipe.parsers.document_parser import DocumentParser, ParsedDocument

# =========================================================================
# ParsedDocument dataclass
# =========================================================================


class TestParsedDocumentDefaults(unittest.TestCase):
    """Test ParsedDocument default field values."""

    def test_default_values(self):
        doc = ParsedDocument(file_path="/tmp/test.pdf", file_type="pdf")
        self.assertEqual(doc.file_path, "/tmp/test.pdf")
        self.assertEqual(doc.file_type, "pdf")
        self.assertEqual(doc.text_content, "")
        self.assertEqual(doc.images, [])
        self.assertEqual(doc.pages, 0)
        self.assertEqual(doc.metadata, {})

    def test_custom_values(self):
        doc = ParsedDocument(
            file_path="/tmp/test.docx",
            file_type="docx",
            text_content="Hello",
            images=[{"data": "abc", "type": "image/png"}],
            pages=5,
            metadata={"author": "Test"},
        )
        self.assertEqual(doc.text_content, "Hello")
        self.assertEqual(len(doc.images), 1)
        self.assertEqual(doc.pages, 5)
        self.assertEqual(doc.metadata["author"], "Test")


class TestParsedDocumentHasImages(unittest.TestCase):
    """Test has_images() method."""

    def test_no_images(self):
        doc = ParsedDocument(file_path="f", file_type="text")
        self.assertFalse(doc.has_images())

    def test_with_images(self):
        doc = ParsedDocument(
            file_path="f",
            file_type="pdf",
            images=[{"data": "base64data", "type": "image/png"}],
        )
        self.assertTrue(doc.has_images())

    def test_multiple_images(self):
        doc = ParsedDocument(
            file_path="f",
            file_type="pdf",
            images=[
                {"data": "a", "type": "image/png"},
                {"data": "b", "type": "image/jpeg"},
            ],
        )
        self.assertTrue(doc.has_images())


class TestParsedDocumentGetFullContent(unittest.TestCase):
    """Test get_full_content() method."""

    def test_text_only(self):
        doc = ParsedDocument(file_path="f", file_type="text", text_content="Hello world")
        self.assertEqual(doc.get_full_content(), "Hello world")

    def test_with_images_appends_placeholder(self):
        doc = ParsedDocument(
            file_path="f",
            file_type="pdf",
            text_content="Some text",
            images=[{"data": "a", "type": "image/png"}, {"data": "b", "type": "image/jpeg"}],
        )
        result = doc.get_full_content()
        self.assertIn("Some text", result)
        self.assertIn("[包含 2 张图片]", result)

    def test_empty_content_no_images(self):
        doc = ParsedDocument(file_path="f", file_type="text", text_content="")
        self.assertEqual(doc.get_full_content(), "")

    def test_empty_content_with_images(self):
        doc = ParsedDocument(
            file_path="f",
            file_type="image",
            text_content="",
            images=[{"data": "x", "type": "image/png"}],
        )
        result = doc.get_full_content()
        self.assertIn("[包含 1 张图片]", result)


# =========================================================================
# DocumentParser.SUPPORTED_EXTENSIONS
# =========================================================================


class TestSupportedExtensions(unittest.TestCase):
    """Test the supported extensions mapping."""

    def test_pdf_extension(self):
        self.assertEqual(DocumentParser.SUPPORTED_EXTENSIONS[".pdf"], "pdf")

    def test_docx_extension(self):
        self.assertEqual(DocumentParser.SUPPORTED_EXTENSIONS[".docx"], "docx")

    def test_doc_extension(self):
        self.assertEqual(DocumentParser.SUPPORTED_EXTENSIONS[".doc"], "docx")

    def test_image_extensions(self):
        for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            self.assertEqual(DocumentParser.SUPPORTED_EXTENSIONS[ext], "image", f"Failed for {ext}")

    def test_text_extensions(self):
        for ext in [".txt", ".md"]:
            self.assertEqual(DocumentParser.SUPPORTED_EXTENSIONS[ext], "text", f"Failed for {ext}")


# =========================================================================
# DocumentParser.parse() — routing and error handling
# =========================================================================


class TestParseRouting(unittest.TestCase):
    """Test parse() method routing and error cases."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self.parser.parse("/nonexistent/path/file.pdf")
        self.assertIn("File not found", str(ctx.exception))

    def test_unsupported_extension(self, tmp_path=None):
        """Unsupported file extension raises ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                self.parser.parse(path)
            self.assertIn("Unsupported file type", str(ctx.exception))
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_pdf")
    def test_routes_pdf(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.pdf", file_type="pdf")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_docx")
    def test_routes_docx(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.docx", file_type="docx")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_docx")
    def test_routes_doc_to_docx_parser(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.doc", file_type="docx")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_image")
    def test_routes_png(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.png", file_type="image")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_image")
    def test_routes_jpg(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.jpg", file_type="image")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_image")
    def test_routes_jpeg(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.jpeg", file_type="image")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_image")
    def test_routes_gif(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.gif", file_type="image")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_image")
    def test_routes_webp(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.webp", file_type="image")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_text")
    def test_routes_txt(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.txt", file_type="text")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(DocumentParser, "_parse_text")
    def test_routes_md(self, mock_parse):
        mock_parse.return_value = ParsedDocument(file_path="t.md", file_type="text")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            self.parser.parse(path)
            mock_parse.assert_called_once_with(path)
        finally:
            os.unlink(path)

    def test_case_insensitive_extension(self):
        """Extension matching should be case-insensitive (Path.suffix.lower())."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".PDF", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            with patch.object(DocumentParser, "_parse_pdf") as mock_parse:
                mock_parse.return_value = ParsedDocument(file_path=path, file_type="pdf")
                self.parser.parse(path)
                mock_parse.assert_called_once()
        finally:
            os.unlink(path)


# =========================================================================
# _parse_pdf() — PyMuPDF (fitz) path
# =========================================================================


class TestParsePdfWithFitz(unittest.TestCase):
    """Test _parse_pdf with PyMuPDF (fitz) available."""

    def setUp(self):
        self.parser = DocumentParser()

    def _make_mock_fitz(self, pages_data, metadata=None, images_per_page=None):
        """Create a mock fitz module.

        Args:
            pages_data: list of str, text content per page
            metadata: dict or None
            images_per_page: list of list of tuples, each tuple is (xref,...)
        """
        mock_fitz = MagicMock()

        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=len(pages_data))
        mock_pdf.metadata = metadata

        mock_pages = []
        for i, text in enumerate(pages_data):
            page = MagicMock()
            page.get_text.return_value = text
            if images_per_page and i < len(images_per_page):
                page.get_images.return_value = images_per_page[i]
            else:
                page.get_images.return_value = []
            mock_pages.append(page)

        mock_pdf.__iter__ = MagicMock(return_value=iter(mock_pages))
        mock_fitz.open.return_value = mock_pdf

        # Setup extract_image
        def extract_image(xref):
            return {
                "image": b"\x89PNG\r\n\x1a\n" + bytes([xref]),
                "ext": "png",
            }

        mock_pdf.extract_image = MagicMock(side_effect=extract_image)

        return mock_fitz, mock_pdf

    @patch.dict("sys.modules", {})
    def test_fitz_basic_text_extraction(self):
        mock_fitz, mock_pdf = self._make_mock_fitz(
            ["Page 1 content", "Page 2 content"],
            metadata={"title": "Test PDF", "author": "Author"},
        )

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertEqual(doc.file_type, "pdf")
        self.assertEqual(doc.pages, 2)
        self.assertIn("Page 1 content", doc.text_content)
        self.assertIn("Page 2 content", doc.text_content)
        self.assertIn("--- 第 1 页 ---", doc.text_content)
        self.assertIn("--- 第 2 页 ---", doc.text_content)
        self.assertEqual(doc.metadata["title"], "Test PDF")
        mock_pdf.close.assert_called_once()

    def test_fitz_with_images(self):
        mock_fitz, mock_pdf = self._make_mock_fitz(
            ["Text on page 1"],
            metadata={},
            images_per_page=[[(42, 0, 0, 0, 0, 0, 0)]],
        )

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertEqual(len(doc.images), 1)
        self.assertEqual(doc.images[0]["type"], "image/png")
        self.assertEqual(doc.images[0]["page"], 1)
        self.assertEqual(doc.images[0]["index"], 0)
        # Verify base64 encoding
        decoded = base64.b64decode(doc.images[0]["data"])
        self.assertTrue(decoded.startswith(b"\x89PNG"))

    def test_fitz_empty_pages_skipped(self):
        mock_fitz, mock_pdf = self._make_mock_fitz(
            ["Content here", "   ", "More content"],
            metadata=None,
        )

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertEqual(doc.pages, 3)
        # Empty page should be skipped in text_content
        self.assertNotIn("第 2 页", doc.text_content)
        self.assertIn("第 1 页", doc.text_content)
        self.assertIn("第 3 页", doc.text_content)

    def test_fitz_none_metadata(self):
        mock_fitz, mock_pdf = self._make_mock_fitz(["text"], metadata=None)

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertEqual(doc.metadata, {})

    def test_fitz_image_extraction_error_handled(self):
        """If image extraction fails, it should be silently skipped."""
        mock_fitz, mock_pdf = self._make_mock_fitz(
            ["page text"],
            metadata={},
            images_per_page=[[(10, 0, 0, 0, 0, 0, 0)]],
        )
        mock_pdf.extract_image.side_effect = Exception("extraction failed")

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        # Image extraction failure should not raise, images list should be empty
        self.assertEqual(len(doc.images), 0)
        self.assertIn("page text", doc.text_content)

    def test_fitz_multiple_images_on_one_page(self):
        mock_fitz, mock_pdf = self._make_mock_fitz(
            ["text"],
            metadata={},
            images_per_page=[[(1, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0)]],
        )

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertEqual(len(doc.images), 2)
        self.assertEqual(doc.images[0]["index"], 0)
        self.assertEqual(doc.images[1]["index"], 1)


# =========================================================================
# _parse_pdf() — pdfplumber fallback
# =========================================================================


class TestParsePdfWithPdfplumber(unittest.TestCase):
    """Test _parse_pdf fallback to pdfplumber when fitz is unavailable."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_pdfplumber_basic_extraction(self):
        mock_pdfplumber = MagicMock()

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 text"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 text"

        mock_pdf_ctx = MagicMock()
        mock_pdf_ctx.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf_ctx)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        def fitz_import_error(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("No fitz")
            if name == "pdfplumber":
                return mock_pdfplumber
            return original_import(name, *args, **kwargs)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        with patch("builtins.__import__", side_effect=fitz_import_error):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertEqual(doc.pages, 2)
        self.assertIn("Page 1 text", doc.text_content)
        self.assertIn("Page 2 text", doc.text_content)
        # pdfplumber does not extract images
        self.assertEqual(len(doc.images), 0)

    def test_pdfplumber_empty_page_skipped(self):
        mock_pdfplumber = MagicMock()

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = None  # empty page

        mock_pdf_ctx = MagicMock()
        mock_pdf_ctx.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf_ctx)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        def fitz_import_error(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("No fitz")
            if name == "pdfplumber":
                return mock_pdfplumber
            return original_import(name, *args, **kwargs)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        with patch("builtins.__import__", side_effect=fitz_import_error):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertNotIn("第 2 页", doc.text_content)


# =========================================================================
# _parse_pdf() — PyPDF2 fallback
# =========================================================================


class TestParsePdfWithPyPDF2(unittest.TestCase):
    """Test _parse_pdf fallback to PyPDF2 when fitz and pdfplumber unavailable."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_pypdf2_basic_extraction(self):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "PyPDF2 page 1"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "PyPDF2 page 2"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        mock_pypdf2_module = types.ModuleType("PyPDF2")
        mock_pypdf2_module.PdfReader = MagicMock(return_value=mock_reader)

        def selective_import(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("No fitz")
            if name == "pdfplumber":
                raise ImportError("No pdfplumber")
            if name == "PyPDF2":
                return mock_pypdf2_module
            return original_import(name, *args, **kwargs)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        with patch("builtins.__import__", side_effect=selective_import):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertEqual(doc.pages, 2)
        self.assertIn("PyPDF2 page 1", doc.text_content)
        self.assertIn("PyPDF2 page 2", doc.text_content)

    def test_pypdf2_empty_page_skipped(self):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = None

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        mock_pypdf2_module = types.ModuleType("PyPDF2")
        mock_pypdf2_module.PdfReader = MagicMock(return_value=mock_reader)

        def selective_import(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("No fitz")
            if name == "pdfplumber":
                raise ImportError("No pdfplumber")
            if name == "PyPDF2":
                return mock_pypdf2_module
            return original_import(name, *args, **kwargs)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        with patch("builtins.__import__", side_effect=selective_import):
            doc = self.parser._parse_pdf("/tmp/test.pdf")

        self.assertNotIn("第 2 页", doc.text_content)


# =========================================================================
# _parse_pdf() — no library available
# =========================================================================


class TestParsePdfNoLibrary(unittest.TestCase):
    """Test _parse_pdf raises ImportError when no PDF library is available."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_no_pdf_library_raises_import_error(self):
        def all_fail_import(name, *args, **kwargs):
            if name in ("fitz", "pdfplumber", "PyPDF2"):
                raise ImportError(f"No {name}")
            return original_import(name, *args, **kwargs)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        with patch("builtins.__import__", side_effect=all_fail_import):
            with self.assertRaises(ImportError) as ctx:
                self.parser._parse_pdf("/tmp/test.pdf")
            self.assertIn("No PDF parser available", str(ctx.exception))
            self.assertIn("PyMuPDF", str(ctx.exception))
            self.assertIn("pdfplumber", str(ctx.exception))
            self.assertIn("PyPDF2", str(ctx.exception))


# =========================================================================
# _parse_docx()
# =========================================================================


class TestParseDocx(unittest.TestCase):
    """Test _parse_docx with mocked python-docx."""

    def setUp(self):
        self.parser = DocumentParser()

    def _make_mock_docx_module(self, paragraphs, tables=None, rels=None):
        """Create a mock docx module.

        Args:
            paragraphs: list of str, paragraph text values
            tables: list of list of list of str, table[row][cell] text
            rels: dict of rel_key -> (target_ref, blob, content_type)
        """
        mock_docx = MagicMock()

        mock_word_doc = MagicMock()

        # Paragraphs
        mock_paragraphs = []
        for text in paragraphs:
            para = MagicMock()
            para.text = text
            mock_paragraphs.append(para)
        mock_word_doc.paragraphs = mock_paragraphs

        # Tables
        mock_tables = []
        if tables:
            for table_data in tables:
                mock_table = MagicMock()
                mock_rows = []
                for row_data in table_data:
                    mock_row = MagicMock()
                    mock_cells = []
                    for cell_text in row_data:
                        cell = MagicMock()
                        cell.text = cell_text
                        mock_cells.append(cell)
                    mock_row.cells = mock_cells
                    mock_rows.append(mock_row)
                mock_table.rows = mock_rows
                mock_tables.append(mock_table)
        mock_word_doc.tables = mock_tables

        # Relations (images)
        mock_rels = {}
        if rels:
            for key, (target_ref, blob, content_type) in rels.items():
                rel = MagicMock()
                rel.target_ref = target_ref
                rel.target_part = MagicMock()
                rel.target_part.blob = blob
                rel.target_part.content_type = content_type
                mock_rels[key] = rel
        mock_word_doc.part.rels = mock_rels

        mock_docx.Document = MagicMock(return_value=mock_word_doc)
        return mock_docx

    def test_basic_paragraph_extraction(self):
        mock_docx = self._make_mock_docx_module(
            paragraphs=["First paragraph", "Second paragraph", "Third paragraph"],
        )

        with patch.dict("sys.modules", {"docx": mock_docx}):
            doc = self.parser._parse_docx("/tmp/test.docx")

        self.assertEqual(doc.file_type, "docx")
        self.assertIn("First paragraph", doc.text_content)
        self.assertIn("Second paragraph", doc.text_content)
        self.assertIn("Third paragraph", doc.text_content)

    def test_empty_paragraphs_skipped(self):
        mock_docx = self._make_mock_docx_module(
            paragraphs=["Content", "   ", "", "More content"],
        )

        with patch.dict("sys.modules", {"docx": mock_docx}):
            doc = self.parser._parse_docx("/tmp/test.docx")

        # Only non-blank paragraphs
        parts = [p for p in doc.text_content.split("\n\n") if p.strip()]
        self.assertEqual(len(parts), 2)

    def test_table_extraction(self):
        mock_docx = self._make_mock_docx_module(
            paragraphs=["Intro text"],
            tables=[
                [
                    ["Header1", "Header2"],
                    ["Cell1", "Cell2"],
                ]
            ],
        )

        with patch.dict("sys.modules", {"docx": mock_docx}):
            doc = self.parser._parse_docx("/tmp/test.docx")

        self.assertIn("Header1 | Header2", doc.text_content)
        self.assertIn("Cell1 | Cell2", doc.text_content)

    def test_page_count_estimate(self):
        """Pages are estimated as len(paragraphs) // 30 + 1."""
        paragraphs = [f"Paragraph {i}" for i in range(65)]
        mock_docx = self._make_mock_docx_module(paragraphs=paragraphs)

        with patch.dict("sys.modules", {"docx": mock_docx}):
            doc = self.parser._parse_docx("/tmp/test.docx")

        self.assertEqual(doc.pages, 65 // 30 + 1)  # 3

    def test_image_extraction(self):
        image_bytes = b"\x89PNG\r\nfake_image_data"
        mock_docx = self._make_mock_docx_module(
            paragraphs=["Text"],
            rels={
                "rId1": ("media/image1.png", image_bytes, "image/png"),
            },
        )

        with patch.dict("sys.modules", {"docx": mock_docx}):
            doc = self.parser._parse_docx("/tmp/test.docx")

        self.assertEqual(len(doc.images), 1)
        self.assertEqual(doc.images[0]["type"], "image/png")
        decoded = base64.b64decode(doc.images[0]["data"])
        self.assertEqual(decoded, image_bytes)

    def test_non_image_rels_ignored(self):
        mock_docx = self._make_mock_docx_module(
            paragraphs=["Text"],
            rels={
                "rId1": ("styles.xml", b"data", "application/xml"),
            },
        )

        with patch.dict("sys.modules", {"docx": mock_docx}):
            doc = self.parser._parse_docx("/tmp/test.docx")

        self.assertEqual(len(doc.images), 0)

    def test_image_extraction_error_silenced(self):
        """If accessing image part fails, it should be silently skipped."""
        mock_docx = MagicMock()
        mock_word_doc = MagicMock()
        mock_word_doc.paragraphs = []
        mock_word_doc.tables = []

        # Create a rel that has "image" in target_ref but raises on target_part access
        broken_rel = MagicMock()
        broken_rel.target_ref = "media/image1.png"
        type(broken_rel).target_part = property(lambda self: (_ for _ in ()).throw(Exception("broken")))
        mock_word_doc.part.rels = {"rId1": broken_rel}

        mock_docx.Document = MagicMock(return_value=mock_word_doc)

        with patch.dict("sys.modules", {"docx": mock_docx}):
            doc = self.parser._parse_docx("/tmp/test.docx")

        # Should not raise, images should be empty
        self.assertEqual(len(doc.images), 0)

    def test_missing_docx_library_raises(self):
        def no_docx_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No docx")
            return original_import(name, *args, **kwargs)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        with patch("builtins.__import__", side_effect=no_docx_import):
            with self.assertRaises(ImportError) as ctx:
                self.parser._parse_docx("/tmp/test.docx")
            self.assertIn("python-docx not installed", str(ctx.exception))


# =========================================================================
# _parse_image()
# =========================================================================


class TestParseImage(unittest.TestCase):
    """Test _parse_image with various image extensions."""

    def setUp(self):
        self.parser = DocumentParser()

    def _test_image_parse(self, suffix, expected_mime):
        import tempfile

        image_data = b"\x89PNG\r\nfake_image_data_for_test"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_data)
            path = f.name
        try:
            doc = self.parser._parse_image(path)
            self.assertEqual(doc.file_type, "image")
            self.assertEqual(doc.pages, 1)
            self.assertEqual(len(doc.images), 1)
            self.assertEqual(doc.images[0]["type"], expected_mime)
            decoded = base64.b64decode(doc.images[0]["data"])
            self.assertEqual(decoded, image_data)
            self.assertIn(os.path.basename(path), doc.text_content)
            self.assertIn("[图片文件:", doc.text_content)
        finally:
            os.unlink(path)

    def test_png(self):
        self._test_image_parse(".png", "image/png")

    def test_jpg(self):
        self._test_image_parse(".jpg", "image/jpeg")

    def test_jpeg(self):
        self._test_image_parse(".jpeg", "image/jpeg")

    def test_gif(self):
        self._test_image_parse(".gif", "image/gif")

    def test_webp(self):
        self._test_image_parse(".webp", "image/webp")

    def test_unknown_extension_defaults_to_png(self):
        """If extension not in mime_map, defaults to image/png."""
        import tempfile

        # Use a trick: call _parse_image directly with an unusual extension
        # The MIME map will fallback to image/png
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"fake_data")
            path = f.name
        try:
            doc = self.parser._parse_image(path)
            self.assertEqual(doc.images[0]["type"], "image/png")
        finally:
            os.unlink(path)

    def test_empty_image_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            doc = self.parser._parse_image(path)
            self.assertEqual(len(doc.images), 1)
            decoded = base64.b64decode(doc.images[0]["data"])
            self.assertEqual(decoded, b"")
        finally:
            os.unlink(path)


# =========================================================================
# _parse_text()
# =========================================================================


class TestParseText(unittest.TestCase):
    """Test _parse_text method."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_basic_text_reading(self):
        import tempfile

        content = "Hello, World!\nThis is a test file.\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            doc = self.parser._parse_text(path)
            self.assertEqual(doc.file_type, "text")
            self.assertEqual(doc.text_content, content)
            self.assertEqual(doc.pages, 1)
            self.assertEqual(doc.images, [])
            self.assertEqual(doc.file_path, path)
        finally:
            os.unlink(path)

    def test_utf8_content(self):
        import tempfile

        content = "Unicode content: \u4e2d\u6587\u6d4b\u8bd5 \U0001f600 \u00e9\u00e0\u00fc"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            doc = self.parser._parse_text(path)
            self.assertEqual(doc.text_content, content)
        finally:
            os.unlink(path)

    def test_empty_text_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            path = f.name
        try:
            doc = self.parser._parse_text(path)
            self.assertEqual(doc.text_content, "")
            self.assertEqual(doc.pages, 1)
        finally:
            os.unlink(path)

    def test_large_text_file(self):
        import tempfile

        content = "Line\n" * 10000
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            doc = self.parser._parse_text(path)
            self.assertEqual(doc.text_content, content)
        finally:
            os.unlink(path)


# =========================================================================
# Integration-style tests via parse()
# =========================================================================


class TestParseIntegration(unittest.TestCase):
    """Integration tests that go through parse() to internal methods."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_parse_text_file_end_to_end(self):
        import tempfile

        content = "Integration test content"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            doc = self.parser.parse(path)
            self.assertEqual(doc.file_type, "text")
            self.assertEqual(doc.text_content, content)
            self.assertEqual(doc.file_path, path)
        finally:
            os.unlink(path)

    def test_parse_image_end_to_end(self):
        import tempfile

        data = b"\x89PNG\r\nfake"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            doc = self.parser.parse(path)
            self.assertEqual(doc.file_type, "image")
            self.assertTrue(doc.has_images())
            self.assertEqual(doc.pages, 1)
        finally:
            os.unlink(path)

    def test_parse_md_file_end_to_end(self):
        import tempfile

        content = "# Heading\n\nParagraph here."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            doc = self.parser.parse(path)
            self.assertEqual(doc.file_type, "text")
            self.assertIn("# Heading", doc.text_content)
        finally:
            os.unlink(path)


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_parsed_document_images_not_shared_between_instances(self):
        """Ensure dataclass default factory creates separate lists."""
        doc1 = ParsedDocument(file_path="a", file_type="pdf")
        doc2 = ParsedDocument(file_path="b", file_type="pdf")
        doc1.images.append({"data": "x", "type": "image/png"})
        self.assertEqual(len(doc2.images), 0)

    def test_parsed_document_metadata_not_shared(self):
        doc1 = ParsedDocument(file_path="a", file_type="pdf")
        doc2 = ParsedDocument(file_path="b", file_type="pdf")
        doc1.metadata["key"] = "value"
        self.assertNotIn("key", doc2.metadata)

    def test_parse_preserves_file_path(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("test")
            path = f.name
        try:
            doc = self.parser.parse(path)
            self.assertEqual(doc.file_path, path)
        finally:
            os.unlink(path)

    def test_constructor_no_args(self):
        """DocumentParser can be instantiated without arguments."""
        parser = DocumentParser()
        self.assertIsInstance(parser, DocumentParser)


if __name__ == "__main__":
    unittest.main()
