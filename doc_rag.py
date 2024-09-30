__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import pdfplumber
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

os.environ["OPENAI_API_KEY"] = api_key

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

# Define the Document class
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

# Function to load PDF documents
def load_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""  # Handle cases where text might be None
                documents.append(Document(content=text, metadata={'filename': filename}))
    return documents

# Load and split documents
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs
'''
# Get answers from the model
def get_answer(query):
    db = Chroma.from_documents(documents=docs, embedding=embeddings)
    chain = load_qa_chain(llm, chain_type="stuff")
    similar_docs = db.similarity_search(query, k=2)  # get two closest chunks
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer
'''

def get_answer(query):
    # Initialize the database and chain
    db = Chroma.from_documents(documents=docs, embedding=embeddings)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Find the most similar documents
    similar_docs = db.similarity_search(query, k=10)  # get two closest chunks
    
    # Prepare the input dictionary for `invoke`
    input_data = {
        'input_documents': similar_docs,
        'question': query
    }
    
    # Invoke the chain and get the result
    result = chain.invoke(input_data)
    
    # Extract and return the 'output_text' from the result dictionary
    return result.get('output_text', 'No output_text found')


# Define the directory containing your PDF documents
directory = "./documents"

# Load PDF documents
documents = load_pdfs(directory)
docs = split_docs(documents)


if __name__ == "__main__":
    # Get user input
    user_query = input("Enter your query: ")

    # Output the user input and the answer
    print("User query:", user_query)
    print("Answer:", get_answer(user_query))
