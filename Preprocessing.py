from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Function to load data from the pdf file

def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs


extracted_data = load_pdf("Data/")

# create chunks of the text.
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = text_split(extracted_data)

