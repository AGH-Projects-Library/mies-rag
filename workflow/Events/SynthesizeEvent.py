# from typing import Dict, List, Any
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

class SynthesizeEvent(Event):
    query: str
    source_nodes: list[NodeWithScore]
    reasoning_nodes: list[NodeWithScore]
