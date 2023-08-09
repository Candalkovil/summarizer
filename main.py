from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from document import load_document
# Load the pretrained summarization model
summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load the Sentence Transformer model
sentence_transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def generate_summary(file_path):
    document = load_document(file_path)
    sentences = document.split('.')
    
    # Remove duplicate sentences
    unique_sentences = list(set(sentences))
    
    # Generate extractive summary using Sentence Transformer
    sentence_embeddings = sentence_transformer_model.encode(unique_sentences, convert_to_tensor=True)
    similarity_matrix = sentence_embeddings @ sentence_embeddings.t()
    
    num_summary_sentences = 3
    selected_indices = similarity_matrix.mean(dim=1).argsort(descending=True)[:num_summary_sentences]
    
    summary_sentences = [unique_sentences[idx] for idx in selected_indices]
    extractive_summary = '. '.join(summary_sentences)
    
    # Generate abstractive summary using BART
    abstractive_input = summarization_tokenizer(extractive_summary, return_tensors="pt", max_length=1024, truncation=True)
    abstractive_output = summarization_model.generate(**abstractive_input, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    final_summary = summarization_tokenizer.decode(abstractive_output[0], skip_special_tokens=True)
    return final_summary


if __name__ == "__main__":
    file_path = 'avengers.txt'  # Change this to the path of your document
    document_summary = generate_summary(file_path)
    print("Generated Summary:")
    print(document_summary)
