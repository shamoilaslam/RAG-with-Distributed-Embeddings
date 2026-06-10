from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from pathlib import Path

def load_docs(data_directory):
    data_path = Path(data_directory)
    print(f"[DEBUG] Data path: {data_path}")
    all_documents = []

    if data_path.is_file() and data_path.suffix.lower() == '.pdf':
        pdf_files = [data_path]
    else:
        pdf_files = list(data_path.glob('**/*.pdf'))
    
    print(f"Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf in pdf_files:
        print(f"\n Processing {pdf.name}")
        try:
            loader = PyMuPDFLoader(pdf)
            documents = loader.load()

            all_documents.extend(documents)
            print(f"\nLoaded {len(documents)} pages")

        except Exception as e:
            print(f"Error {e}")
    return all_documents

