"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from dotenv import load_dotenv
load_dotenv()
# Here we load in the data in the format that Notion exports it in.
ps = list(Path("Notion_DB/").glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p, encoding='UTF8') as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n","."],
    chunk_size=1500,
    chunk_overlap=100)
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
