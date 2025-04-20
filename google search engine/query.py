import argparse
import os
from dotenv import load_dotenv
from pdfsearch import get_manual_pdf, extract_text_from_pdf
from bs4 import BeautifulSoup

load_dotenv()

def try_decode_fake_pdf_as_html(pdf_path: str) -> str | None:
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()
        if content.lstrip().startswith(b"<!DOCTYPE html") or b"<html" in content[:500].lower():
            print("⚠️ File appears to be an HTML page, not a real PDF.")
            soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n").strip()
            output_path = "others/fallback_extracted_text.txt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(text)
            print(f"✅ Extracted HTML text saved to {output_path}")
            return text
    except Exception as e:
        print(f"❌ Error decoding file: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Search, download a PDF manual, and extract its text"
    )
    parser.add_argument("product_name", help="Name of the product to search for")
    parser.add_argument(
        "--output", "-o", default="manual.txt",
        help="File path to save the extracted text"
    )
    parser.add_argument(
        "--temp-pdf", default="others/manual.pdf",
        help="Temporary path to save the downloaded PDF"
    )
    args = parser.parse_args()

    pdf_path = get_manual_pdf(args.product_name, save_path=args.temp_pdf)
    if not pdf_path:
        print(f"❌ No PDF downloaded for '{args.product_name}'")
        return
    print(f"📥 PDF downloaded to: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("⚠️ No text extracted, attempting HTML fallback...")
        fallback = try_decode_fake_pdf_as_html(pdf_path)
        if fallback:
            text = fallback

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Extracted text saved to: {args.output}")

if __name__ == "__main__":
    main()
