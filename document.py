import PyPDF2

def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.txt':
        with open(file_path, 'r') as txt_file:
            return txt_file.read()
    elif file_extension.lower() == '.pdf':
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return ' '.join([page.extract_text() for page in pdf_reader.pages])
    else:
        raise ValueError("Unsupported file format. Supported formats: .txt, .pdf")
