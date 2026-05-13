import pdfplumber

with pdfplumber.open('temp_uploads/Full Proposal (3).pdf') as pdf:
    page = pdf.pages[0]
    
    print('=== TEXT ===')
    print(page.extract_text())
    
    print('\n=== TABLES ===')
    tables = page.extract_tables()
    print(f'Found {len(tables)} tables')
    
    for i, t in enumerate(tables):
        print(f'\n--- Table {i+1} ---')
        for row in t:
            print(row)