#Setup logging diagnostics
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#Import all necessary modules
from llama_index import (ServiceContext, download_loader, KnowledgeGraphIndex)
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from pyvis.network import Network
from Ipython.display import display, HTML
from dotenv import load_dotenv
import Ipython
import os

#Set up LLM and Embedding Model
load_dotenv("llama.env")
face_token = os.getenv("FACE_TOKEN")
llm = HuggingFaceInferenceAPI(model_name = "HuggingFaceH4/zephyr-7b-beta", face_token=face_token)
embed_model = LangchainEmbedding(HuggingFaceInferenceAPIEmbeddings(model_name = 
                                                                   "HuggingFaceH4/zephyr-7b-beta", 
                                                                   face_token=face_token))
#Load data
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Tesla Cybertruck'])

#Create Knowledge Graph Index
service_context = ServiceContext.from_defaults(chunk_size=300, chunk_overlap=30,
                                               llm=llm, embed_model=embed_model)
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(documents=documents, max_triplets_per_chunk=3, 
                                           service_context=service_context,
                                           storage_context=storage_context, 
                                           include_embeddings=True)

#Set up Graph Visualization and Display
graph = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources = "in_line", directed=True)
net.from_nx(graph)
net.show("graph.html")

#Create Query Engine
query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize",
                                     embedding_mode="hybrid", similarity_top_k=5)
response = query_engine.query("According to Musk, what inspired the design of the Cybertruck?")
print(response)

