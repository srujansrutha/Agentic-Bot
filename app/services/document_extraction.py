import io

from docx import Document as DocxDocument
from pypdf import PdfReader


def extract_content(filename: str, file_bytes: bytes) -> tuple[str, list[bytes]]:
    """Detects format from the file's extension and returns
    (extracted_text, list_of_embedded_image_bytes)."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension == "pdf":
        return _extract_pdf(file_bytes)
    if extension == "docx":
        return _extract_docx(file_bytes)
    if extension == "txt":
        return file_bytes.decode("utf-8", errors="ignore"), []

    raise ValueError(f"Unsupported file type: .{extension or '(none)'} — use .pdf, .docx, or .txt")


def _extract_pdf(file_bytes: bytes) -> tuple[str, list[bytes]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts: list[str] = []
    images: list[bytes] = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
        images.extend(img.data for img in page.images)
    return "\n\n".join(text_parts), images


def _extract_docx(file_bytes: bytes) -> tuple[str, list[bytes]]:
    doc = DocxDocument(io.BytesIO(file_bytes))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    images = [rel.target_part.blob for rel in doc.part.rels.values() if "image" in rel.reltype]
    return text, images
