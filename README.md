# README for Knowledge Graph Index Project

This project uses various libraries to create a Knowledge Graph Index 
from a set of documents, visualize the graph, and query it.

## Functionality
The code sets up logging diagnostics, imports necessary modules, 
sets up Language Model (LLM) and Embedding Model, loads data, creates a Knowledge Graph Index, 
sets up graph visualization and display, and creates a Query Engine.

## Usage
The code is used to create a Knowledge Graph Index from a set of documents 
(in this case, the Wikipedia page for 'Tesla Cybertruck'), visualize the graph, and query it.
The query in this example is "According to Musk, what inspired the design of the Cybertruck?".

## Limitations
The code is dependent on the availability and functionality of the imported modules.
The quality of the Knowledge Graph Index and the accuracy of the query results depend on the quality and relevance of the input documents.

## Modules/Libraries Needed
The code uses several modules including `logging`, `sys`, `os`, `Ipython`, `dotenv`, `pyvis.network`, 
and several modules from `llama_index` and `langchain.embeddings`. 
Please ensure these are installed and available in your Python environment.
