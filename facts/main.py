from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search(
    "What is an interesting fact about the English language?"
)

for result in results:
    print("\n")
    print(result.page_content)
