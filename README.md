# AI Exam Prep Coach

A Streamlit-based application that helps students prepare for exams by processing PDF documents and providing AI-powered study assistance.

## Features

- **PDF Processing**: Upload and process PDF documents to extract text content
- **Local Vector Search**: Uses FAISS and HuggingFace embeddings for semantic search (no API calls for embeddings)
- **AI-Powered Responses**: Uses Google Gemini for generating answers, explanations, quizzes, and notes
  - Ask Questions: Get answers from the uploaded content
  - Explain Topics: Simplified explanations for exam preparation
  - Generate Quiz: Create multiple-choice questions with answers
  - Make Notes: Generate bullet-point revision notes
  - Quick Revision: Summarized revision sheets
- **Chat History**: Keep track of your study sessions

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-exam-prep-coach
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`
   - Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload PDF files using the sidebar

4. Click "Process" to create the knowledge base

5. Select a study mode and enter your query

## Requirements

- Python 3.8+
- Google Gemini API key
- Internet connection for API calls

## Dependencies

- streamlit
- google-generativeai
- python-dotenv
- langchain-core
- langchain-community
- langchain-text-splitters
- PyPDF2
- faiss-cpu
- sentence-transformers

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── chatpdf.py            # Alternative implementation (legacy)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── faiss_index/          # Vector database storage
│   └── index.faiss
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.# Ai-Exam-Preparation-Coach
