import os

# openai/groq
# evaluation works only with openai
API = "openai"
# API = "groq"

MODEL = "gpt-4o-mini" 
# MODEL = "llama-3.1-8b-instant" 
# MODEL = "llama3-8b-8192" 
# MODEL = "llama-3.2-1b-preview"
# MODEL = "mixtral-8x7b-32768"
# MODEL = "gemma2-9b-it"

BASE_DIR = './'

INPUT_PATH = os.path.join(BASE_DIR, 'input')  
OUTPUT_PATH = os.path.join(BASE_DIR, 'output')  
STORAGE_PATH = os.path.join(BASE_DIR, 'storage', API)

# Maximum number of iterations 
MAX_STEPS = 3

# Disable the second loop (subquestions)
DESABLE_SECOND_LOOP = False

# Enable evaluation of results
# Determines whether to evaluate the results
EVALUATION = False  

# Use Ragas library for evaluation if EVALUATION = True
# Activates Ragas for evaluation when EVALUATION is enabled
RAGAS = False  

# Use Geval library for evaluation if EVALUATION = True
# Activates Geval for evaluation when EVALUATION is enabled
G_EVAL = False

GROUND_TRUTH = True

# Clear the storage folder before starting
# If set to True, clears the storage folder to ensure a clean run without prior data
CLEAR_STORAGE = False

# Use the Cohere reranker for better context selection
# If True, utilizes the Cohere reranker to improve the selection of the best context
COHERE_RERANK = False