from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

c = canvas.Canvas('data/documents/test_agriculture.pdf', pagesize=letter)
c.drawString(100, 750, 'Agriculture Guide')
c.drawString(100, 700, 'Chapter 1: Basics of Farming')
c.drawString(100, 650, 'Farming is the cultivation of crops and raising of livestock.')
c.drawString(100, 600, 'Wheat is a major staple crop grown worldwide.')
c.drawString(100, 550, 'Rice requires more water than other grains.')
c.drawString(100, 500, 'Irrigation is crucial for crop production.')
c.save()
print("Test PDF created successfully")
