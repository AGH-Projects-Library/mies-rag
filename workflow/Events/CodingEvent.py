from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Event

class CodingEvent(Event):
    question: str
    answer: str
    source_nodes: list[NodeWithScore]