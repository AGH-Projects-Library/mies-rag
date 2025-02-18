from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
# from ragas.dataset_schema import EvaluationResult

class AnswerEvent(Event):
    question: str
    answer: str
    source_nodes: list[NodeWithScore]
