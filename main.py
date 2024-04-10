from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import openai
from langchain.llms import OpenAI
from langchain import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

load_dotenv()


# Function to load data from the pdf file

# def load_pdf(data):
#     loader = DirectoryLoader(data,
#                              glob="*.pdf",
#                              loader_cls=PyPDFLoader)
#     docs = loader.load()
#     return docs
#
#
# extracted_data = load_pdf("Data/")


# create chunks of the text.

# def text_split(extracted_data):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=20,
#     )
#     text_chunks = text_splitter.split_documents(extracted_data)
#     return text_chunks
#
#
# text_chunks = text_split(extracted_data)


# print(text_chunks[567].page_content.replace("\n", " "))

#includes the embeddings model used to create embeddings of the textual data

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


embeddings = download_hugging_face_embeddings()

'''
for i in range(0, len(text_chunks)):
    temp = {
        'id': None,
        'metadata': None,
        'values': None
    }
    temp_for_metadata = {}
    txt = text_chunks[i].page_content
    id = str(i)
    embeddings = model.encode(txt)
    temp['id'] = id
    temp_for_metadata['page_content'] = txt
    temp['metadata'] = temp_for_metadata
    temp['values'] = embeddings
    vector_data.append(temp)

print(vector_data)
'''

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# pinecone is used here to store data to the pinecone database.

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "chatbot"

pc = Pinecone(os.getenv("PINECONE_API_KEY"))
index = pc.Index("chatbot")

# docsearch = PineconeVectorStore.from_documents(
#     text_chunks,
#     embeddings,
#     index_name = index_name
# )


docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)


def retrieve_query(query, k=3):
    matching_results = docsearch.similarity_search(query, k=k)
    return matching_results

llm = OpenAI(temperature=0.5)
chain = load_qa_with_sources_chain(llm, chain_type="stuff")


def retrieve_answers(query):
    doc_search = retrieve_query(query)
    response = chain.run(question = query, input_documents=doc_search)
    return response


# our_query = "what is atopic dermatis"
# answer = retrieve_answers(our_query)
# print(answer)

while True:
    user_input = input(f"Input Prompt: ")
    result = retrieve_answers(user_input)
    print("Response: " ,result)

