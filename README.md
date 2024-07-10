# Keyword Extraction from Images and PDFs

This Python script extracts keywords from images and PDFs, translates non-English text to English (if applicable), and generates a PDF report with extracted keywords.

## Setup Instructions

### Clone the Repository

```bash
git clone <repository_url>
cd keywords  # Navigate to the project directory
# Create and activate virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
python3 -m nltk.downloader punkt
python3 -m nltk.downloader stopwordspython3 keywords.py

