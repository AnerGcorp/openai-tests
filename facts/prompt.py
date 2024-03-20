from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)
retriever = RedundantFilterRetriever(
    chroma=db,
    embeddings=embeddings
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.invoke("What is an interesting fact about the English language?")

print(result)
