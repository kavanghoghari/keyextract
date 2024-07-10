import os
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fpdf import FPDF

# Set up Tesseract path (update this path as needed)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Ensure the required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Supported languages for translation
supported_langs = ['fr', 'es', 'it', 'pt', 'ro', 'de', 'nl', 'sv', 'fi', 'da', 'no', 'pl', 'gu', 'hi']

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(image_path, lang='guj+hin+eng')

def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang='guj+hin+eng')
    return text

def translate_text(text, src_lang, target_lang='en'):
    try:
        if src_lang in supported_langs:
            model_name = f'Helsinki-NLP/opus-mt-{src_lang}-en'
        else:
            print(f"Translation for language '{src_lang}' is not supported.")
            return text

        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    except OSError as e:
        print(f"Translation error: {e}")
        print(f"Translation for language '{src_lang}' is not available.")
        return text
    
    except Exception as e:
        print(f"Error occurred during translation: {e}")
        return text

def filter_keywords(text):
    word_tokens = word_tokenize(text)
    filtered_keywords = [word for word in word_tokens if word.isalnum() and word.lower() not in stop_words]
    return filtered_keywords

def process_files(folder_path):
    result = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_path)
        elif filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            continue
        
        if not text:
            continue

        # Detect language and translate if necessary
        try:
            detected_lang = detect(text)
            if detected_lang != 'en':
                text = translate_text(text, detected_lang, 'en')
        except Exception as e:
            print(f"Language detection error: {e}")
        
        keywords = filter_keywords(text)
        result[filename] = keywords

    return result

def save_to_pdf(result, output_pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for filename, keywords in result.items():
        pdf.cell(200, 10, txt=filename, ln=True)
        pdf.multi_cell(0, 10, txt=", ".join(keywords))

    pdf.output(output_pdf_path)

if __name__ == "__main__":
    folder_path = 'images'  # Update this path as needed
    output_pdf_path = 'output.pdf'  # Specify the output PDF file path
    
    # Process files in the specified folder
    result = process_files(folder_path)
    
    # Save results to PDF
    save_to_pdf(result, output_pdf_path)
