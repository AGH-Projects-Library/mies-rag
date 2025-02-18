import os
from llama_index.llms.openai import OpenAI 
from llama_parse import LlamaParse 
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core import SimpleDirectoryReader

class VectorQueryEngineCreator:
    def __init__(self, llama_parse_api_key, cohere_api_key, model, input_path, storage_path, cohere_rerank, api):
        self.llama_parse_api_key = llama_parse_api_key
        self.cohere_api_key = cohere_api_key
        self.model = model
        self.input_path = input_path
        self.storage_path = storage_path
        self.cohere_rerank = cohere_rerank
        self.api = api

    def parse_pdf_to_nodes(self, path_to_pdf):
        # todo scipdf_parser or some other pdf parser
        if self.api == "openai":
            documents = LlamaParse(
                api_key=self.llama_parse_api_key,
                result_type="markdown"
            ).load_data(path_to_pdf)
            node_parser = MarkdownElementNodeParser(
                llm=OpenAI(model=self.model), num_workers=8
            )
            nodes = node_parser.get_nodes_from_documents(documents)
        elif self.api == "groq":
            documents = SimpleDirectoryReader(input_files=[path_to_pdf]).load_data()
            node_parser = None
            nodes = None
            
        # Remove chapters containing references
        documents = [doc for doc in documents if 'references' not in doc.text.lower()]

        return documents, node_parser, nodes

    def create_vector_index(self, documents, node_parser, nodes):
        if self.api == "openai":
            base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
            vector_index = VectorStoreIndex(base_nodes + objects)
        elif self.api == "groq":
            vector_index = VectorStoreIndex.from_documents(documents)
        return vector_index

    def create_vector_query_engine(self, vector_index):
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=5,
        )
        response_synthesizer = get_response_synthesizer()

        if self.cohere_rerank:
            os.environ["COHERE_API_KEY"] = self.cohere_api_key
            cohere_api_key = os.environ["COHERE_API_KEY"]
            cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=5)

            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_mode='tree_summarize',
                response_synthesizer=response_synthesizer,
                node_postprocessors=[cohere_rerank]
            )
        else:
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_mode='tree_summarize',
                response_synthesizer=response_synthesizer,
            )
        return query_engine

    def get_query_engine(self, file):
        vector_index_persist_path = f'{self.storage_path}/{file}_vector_index/'

        if os.path.exists(vector_index_persist_path):
            storage_context = StorageContext.from_defaults(persist_dir=vector_index_persist_path)
            vector_index = load_index_from_storage(storage_context)
        else:
            pdf_path = os.path.join(self.input_path, f"{file}.pdf")
            documents, node_parser, nodes = self.parse_pdf_to_nodes(pdf_path)
            vector_index = self.create_vector_index(documents, node_parser, nodes)
            vector_index.storage_context.persist(vector_index_persist_path)

        query_engine = self.create_vector_query_engine(vector_index)
        return query_engine
