from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.multi_cell(0, 10, txt="Page 1: This is the first page of a sample PDF.\n"
                           "It is used to test the OCR logging.")

pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.multi_cell(0, 10, txt="Page 2: The OCR should extract this text as well.\n"
                           "Make sure both pages are processed.")

pdf.output("pdf_in/test_two_pages.pdf")
