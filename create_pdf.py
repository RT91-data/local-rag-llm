# create_pdf.py
from fpdf import FPDF
import os

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Sample IT service contract text
contract_text = """
IT SERVICE AGREEMENT

1. Scope of Services
The Service Provider shall provide IT support and software maintenance services to the Client, including installation, configuration, troubleshooting, and updates.

2. Obligations of the Client
The Client shall provide all necessary access, information, and approvals to allow the Service Provider to perform services efficiently.

3. Payment Terms
The Client agrees to pay the Service Provider $5000 per month, due within 15 days of invoice receipt.

4. Confidentiality
Both parties agree to maintain confidentiality of sensitive information exchanged during the term of this Agreement.

5. Termination
Either party may terminate this Agreement with 30 days written notice.
"""

# Add text to PDF
for line in contract_text.split("\n"):
    pdf.multi_cell(0, 8, line)

# Save PDF
pdf.output("data/sample.pdf")

print("sample.pdf created successfully in data/ folder!")