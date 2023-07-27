import streamlit as st
import os
import pdfplumber
import requests
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

# Set OpenAI API key
openai_api_key = "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"

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Download PDFs and extract text
def download_and_extract_pdfs(paper_urls):
    for i, url in enumerate(paper_urls):
        response = requests.get(url)
        if response.status_code == 200:
            pdf_filename = f"Paper{i+1}.pdf"
            with open(pdf_filename, 'wb') as file:
                file.write(response.content)
            paper_text = extract_text_from_pdf(pdf_filename)
            text_filename = f"Paper{i+1}.txt"
            save_text_to_file(paper_text, text_filename)

# Function to save the extracted content to a text file
def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


def setup():
    # URLs to the PDF files
    paper_urls = [
        "https://dl.acm.org/doi/pdf/10.1145/3397271.3401075",
        "https://arxiv.org/pdf/2104.07186.pdf",
        "https://arxiv.org/pdf/2106.14807.pdf",
        "https://arxiv.org/pdf/2301.03266.pdf",
        "https://arxiv.org/pdf/2303.07678.pdf"
    ]
    download_and_extract_pdfs(paper_urls)

    # Initialize the qa model
    openai_api_key = os.getenv("OPENAI_API_KEY", "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")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    loader = DirectoryLoader('research_paper', glob="**/*.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vecstore = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type='stuff',
        retriever=vecstore.as_retriever()
    )

    return qa

def query(q):
    answer = qa.run(q)
    return answer

if __name__ == "__main__":
    # Setup the qa model
    qa = setup()
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            color: #ADD8E6;
            text-align: center;
            margin-bottom: 30px;
        }
        .question-input {
            font-size: 18px;
            border: 1px solid #ccc;
            padding: 8px;
            width: 80%;
            margin: 0 auto;
            margin-bottom: 20px;
        }
        .answer {
            font-size: 16px;
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            padding: 10px;
            color: #333; /* Set the text color to black */
        }
        .get-answer-btn {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Custom title
    st.markdown("<h1 class='title'>FARMWISE AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 align='center'>Research Paper Analysis</h2>", unsafe_allow_html=True)

    # Your question input section
    question = st.text_input("Enter your question:", "")

    # Display the answer
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
        else:
            # Call the query function to get the answer
            answer = query(question)

            # Display the answer
            st.subheader("Answer:")
            st.write(str(answer))  # Convert the answer to a string before displaying

