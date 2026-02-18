import fitz
import sys

def diagnose(pdf_path):
    print(f"Diagnosing: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        print(f"Pages: {len(doc)}")
        
        text_content = ""
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text_content += text
            if i < 3:
                print(f"--- Page {i+1} Sample ---")
                print(text[:500])
                print("-----------------------")
        
        if not text_content.strip():
            print("RESULT: SCANNED IMAGE (No Text Layer)")
        else:
            print(f"RESULT: TEXT LAYER FOUND ({len(text_content)} chars)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose(sys.argv[1])
