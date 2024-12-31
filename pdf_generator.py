from fpdf import FPDF

class PDFGenerator(FPDF):
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, image_path):
        self.image(image_path, x=10, y=None, w=190)
        self.ln(10)
