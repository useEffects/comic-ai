from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain import PromptTemplate
import textwrap


def load_csv_data(file_path):
    loader = CSVLoader(
        file_path=file_path,
        csv_args={"delimiter": ",", "quotechar": '"', "fieldnames": ["Prompt"]},
    )
    return loader.load()


def split_docs(documents, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks


def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": normalize_embedding},
    )


def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore


def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def get_response(query, chain):
    response = chain({"query": query})
    wrapped_text = textwrap.fill(response["result"], width=100)
    print(wrapped_text)


llm = Ollama(model="llama2", temperature=0)
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

docs = load_csv_data(file_path="lib/datasets/prompts.csv")
documents = split_docs(documents=docs)

vectorstore = create_embeddings(documents, embed)
retriever = vectorstore.as_retriever()





