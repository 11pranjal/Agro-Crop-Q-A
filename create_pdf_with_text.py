import io
from PyPDF2 import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Create a PDF with text using reportlab
buffer = io.BytesIO()
c = canvas.Canvas(buffer, pagesize=letter)
c.drawString(100, 750, "Agricultural Guide")
c.drawString(100, 700, "Chapter 1: Introduction to Farming")
c.drawString(100, 650, "Farming is the practice of cultivating plants and raising animals.")
c.drawString(100, 600, "Wheat is one of the most important crops worldwide.")
c.drawString(100, 550, "Rice farming requires careful water management.")
c.drawString(100, 500, "Irrigation systems are crucial for crop success.")
c.save()

# Write the PDF to file
buffer.seek(0)
with open('data/documents/agriculture.pdf', 'wb') as f:
    f.write(buffer.getvalue())

print("Agriculture PDF created successfully with text content")
