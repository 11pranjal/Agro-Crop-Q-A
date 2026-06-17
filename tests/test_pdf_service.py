from src.services.pdf_service import PDFService
from io import BytesIO


def test_process_empty_pdf():
    svc = PDFService()
    stream = BytesIO(b"")
    result = svc.process_pdf(stream)
    assert isinstance(result, dict)
    assert result.get("success") is False
