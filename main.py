import os
import shutil
import time
import asyncio
import nest_asyncio
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI 
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.utils.workflow import draw_all_possible_flows


from config.config import (
    API,
    INPUT_PATH, 
    OUTPUT_PATH, 
    STORAGE_PATH, 
    MODEL, 
    MAX_STEPS, 
    DESABLE_SECOND_LOOP,
    EVALUATION, 
    RAGAS,
    G_EVAL,
    GROUND_TRUTH,
    CLEAR_STORAGE, 
    COHERE_RERANK,
)
from config.config_keys import (
    OPENAI_API_KEY, 
    LLAMA_PARSE_API_KEY, 
    COHERE_API_KEY,
    GROQ_API_KEY
)

from config.queries import QUERIES
from config.ground_truth import GROUND_TRUTH_LIST

from utils.QuestionsManager import QuestionsManager
from utils.ReportGenerator import ReportGenerator
from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
from utils.RAGEvaluator import RAGEvaluator
from workflow.MultiStepQueryEngineWorkflow import MultiStepQueryEngineWorkflow


def main():
    start = time.time()

    nest_asyncio.apply()
    
    if API == "openai":
        print("Using GPT-4o Mini model")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        llm = OpenAI(model=MODEL)
        Settings.llm = llm

    elif API == "groq":
        print(f"Using {MODEL} model")
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        llm = Groq(model=MODEL)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = llm
        Settings.embed_model = embed_model
    

    if CLEAR_STORAGE:
        for item in os.listdir(STORAGE_PATH):
            item_path = os.path.join(STORAGE_PATH, item)
            if os.path.basename(item_path) == ".gitkeep":
                continue
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
    output_path = os.path.join(OUTPUT_PATH, f"{time.strftime("%Y.%m.%d_%H.%M.%S")}")
    
    questionsManager = QuestionsManager(QUERIES, STORAGE_PATH, Settings.llm)
    raportGenerator = ReportGenerator(QUERIES, output_path)

    files = os.listdir(INPUT_PATH)
    pdf_files = []

    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.splitext(file)[0])

    for i, file in enumerate(pdf_files):
        
        query_engine = VectorQueryEngineCreator(LLAMA_PARSE_API_KEY, COHERE_API_KEY, MODEL, INPUT_PATH, STORAGE_PATH, COHERE_RERANK, API).get_query_engine(file)
        workflow = MultiStepQueryEngineWorkflow(timeout=1000)
        result = asyncio.run(process_file(f"[{i+1}/{len(pdf_files)}]", file, workflow, questionsManager, Settings.llm, query_engine))
        info = {} # todo: get info from the pdf file (author, title, etc.)
        raportGenerator.generate_partial_report(file, info, result)
        
    raportGenerator.generate_main_report()
    end = time.time()
    execution_time = end - start
    raportGenerator.generate_config_report(execution_time)
    
    print("END")
    print(f"Execution time: {execution_time} seconds")
    return 

async def process_file(f, filename, workflow, questionsManager, llm, query_engine):
    results = []
    for i in range(questionsManager.count):
        print(f"\nProcessing: file {f}; query [{i+1}/{questionsManager.count}]")
        result = await workflow.run(
            llm = llm,
            query = questionsManager.get_question(i),
            query_engine = query_engine,
            max_steps = MAX_STEPS,
            disable_second_loop = DESABLE_SECOND_LOOP,
        )
        results.append(result)
    
    if EVALUATION:
        if GROUND_TRUTH:
            ground_truth = GROUND_TRUTH_LIST[filename]
        else:
            ground_truth = ["" for i in range(questionsManager.count)]
        evaluation = RAGEvaluator(questionsManager.get_questions(), llm, results, MODEL, RAGAS, G_EVAL, ground_truth, filename).evaluate()
    else:
        evaluation = None

    for i, r in enumerate(results):
        if "contexts" in r:
            del r["contexts"]
        if EVALUATION:
            r["evaluation"] = evaluation[i]
        else:
            r["evaluation"] = None
    
    return results

main()