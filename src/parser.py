import os
import pymupdf4llm
from docx import Document

class ResumeParser:
    def __init__(self):
        pass

    def parse_file(self, file_path: str) -> str:
        """
        Parses a PDF or DOCX file and returns its content as a Markdown string.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext == '.docx':
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _parse_pdf(self, file_path: str) -> str:
        """
        Uses pymupdf4llm for high-fidelity markdown extraction.
        """
        try:
            md_text = pymupdf4llm.to_markdown(file_path)
            return md_text
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return ""

    def _parse_docx(self, file_path: str) -> str:
        """
        Uses python-docx to extract text from DOCX files.
        """
        try:
            doc = Document(file_path)
            # Basic markdown conversion (could be enhanced for tables, etc.)
            md_text = ""
            for para in doc.paragraphs:
                if para.style.name.startswith('Heading'):
                    level = min(int(para.style.name.split(' ')[1]), 6) if len(para.style.name.split(' ')) > 1 else 1
                    md_text += f"{'#' * level} {para.text}\n\n"
                else:
                    md_text += f"{para.text}\n\n"
            return md_text
        except Exception as e:
            print(f"Error parsing DOCX {file_path}: {e}")
            return ""

if __name__ == "__main__":
    # Simple test execution if run directly
    parser = ResumeParser()
    print("Parser initialized.")
