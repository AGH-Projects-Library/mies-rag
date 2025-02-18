from llama_index.core.workflow import Event

class QueryEvent(Event):
    query: str

