import json
from llama_index.core import get_response_synthesizer
from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step
)

from .Events.SetupEvent import SetupEvent
from .Events.SubquestionEvent import SubquestionEvent
from .Events.QueryEvent import QueryEvent
from .Events.AnswerEvent import AnswerEvent
from .Events.SynthesizeEvent import SynthesizeEvent
from .Events.CodingEvent import CodingEvent

class MultiStepQueryEngineWorkflow(Workflow):
    def refine_question(self, llm, query, question, reasoning) -> str:
        prompt = f'''
            Your task is to reformulate a given question to better reflect the collected information while staying aligned with the main topic. The original question was crafted based on global information about the topic and should retain its core essence but become more precise and tailored to the new data.

            ### Input:
            - **Original question:** {question}
            - **Topic:** {query["topic"]}
            - **Collected information:**
            {f"\n\t".join([f'{i}. {r["answer"].strip()}' for i, r in enumerate(reasoning)])}

            ### Task:
            Reformulate the question to:
            1. Stay consistent with the provided topic ({query["topic"]}).
            2. Incorporate key aspects of the collected information.
            3. Remain clear and specific.

            Provide only the reformulated question as the output. Do not include any explanations, formatting, or additional text. The answer must be a single plain string.

        '''

        new_query = llm.complete(prompt)
        
        return f"{new_query!s}"

    @step(pass_context = True)
    async def setup_workflow(self, ctx: Context, ev: StartEvent ) -> SetupEvent:
        if not hasattr(ctx.data, "llm"):
            ctx.data["llm"] = ev.llm

        if not hasattr(ctx.data, "query_engine"):
            ctx.data["query_engine"] = ev.query_engine

        ctx.data["query"] = ev.query["query"]
        ctx.data["question"] = ev.query["question"]
        ctx.data["source_nodes_dict"] = {}
        ctx.data["question_count"] = 0
        ctx.data["reasoning"] = []
        ctx.data["max_steps"] = ev.max_steps
        ctx.data["cur_steps"] = 0
        ctx.data["question_collect_count"] = 0
        ctx.data["disable_second_loop"] = ev.disable_second_loop
        ctx.data["should_synthesize"] = False
        
        return SetupEvent()
            
    @step(pass_context=True)
    async def query_multi_step(self, ctx: Context, ev: SetupEvent | AnswerEvent ) -> SubquestionEvent | QueryEvent | SynthesizeEvent:
        
        # [LOOP 1] Collect reasoning nodes for the start question
        if isinstance(ev, SetupEvent):
            ctx.data["question_collect_count"] = 1
            return QueryEvent(query = ctx.data["question"], sub = False)
        
        new_node = 0
        ready = ctx.collect_events(ev, [AnswerEvent]*ctx.data["question_collect_count"])
        pref = "[MQ]" if ctx.data["question_collect_count"]==1 else "[SQ]"
        if ready is None:
            return None
        for event in ready:
            for node in event.source_nodes:
                if node.node_id not in ctx.data["source_nodes_dict"]:
                    ctx.data["source_nodes_dict"][node.node_id] = node
                    new_node += 1
                else:
                    ctx.data["source_nodes_dict"][node.node_id].score += node.score  
            ctx.data["reasoning"].append({"question": f"{pref} {event.question}", "answer": event.answer})
        print(f"{"[MQ]" if ctx.data["question_collect_count"]==1 else "[SQ]"} New nodes (contexts): {new_node}")
        if ctx.data["cur_steps"] >= ctx.data["max_steps"] or new_node == 0:
            ctx.data["should_synthesize"] = True
        # disable second loop [config option] (without subquestions)
        elif ctx.data["disable_second_loop"]:
            ctx.data["cur_steps"] += 1
            new_question = self.refine_question(ctx.data["llm"], ctx.data["query"], ctx.data["question"], ctx.data["reasoning"])
            ctx.data["question"] = new_question
            ctx.data["question_collect_count"] = 1
            return QueryEvent(query = ctx.data["question"])
        else:
            # [LOOP 1] refine question -> new question
            if ctx.data["question_collect_count"] == 3:
                ctx.data["cur_steps"] += 1
                new_question = self.refine_question(ctx.data["llm"], ctx.data["query"], ctx.data["question"], ctx.data["reasoning"])
                ctx.data["question"] = new_question
                ctx.data["question_collect_count"] = 1
                return QueryEvent(query = ctx.data["question"])
            # [LOOP 2] subquestion: collect more reasoning for the current question
            else:
                ctx.data["question_collect_count"] = 3
                return SubquestionEvent()
            
        # [synthesize] the final response
        if ctx.data["should_synthesize"]:
            # normalize the score of source nodes
            source_nodes = []
            for node in ctx.data["source_nodes_dict"].values():
                node.score = node.score / ctx.data["question_count"]
                source_nodes.append(node)

            # create reasoning nodes
            text_chunks = [(f"\nQuestion: {r['question']}\n"f"Answer: {r['answer']}") for r in ctx.data["reasoning"]]
            reasoning_nodes = [
                NodeWithScore(node=TextNode(text = text_chunk))
                for text_chunk in text_chunks
            ]
            return SynthesizeEvent(
                query = ctx.data["question"],
                source_nodes = source_nodes,
                reasoning_nodes = reasoning_nodes
            )    
    
    @step(pass_context=True)
    async def subquestion(self, ctx: Context, ev: SubquestionEvent  ) -> QueryEvent:
        
        question = ctx.data["question"]
        reasoning = ctx.data["reasoning"]
        prompt = f"""
            Based on the main question: "{question}", and the history of previous questions and answers provided below:
            History:\n
            {reasoning}
            \n
            Generate 3 specific and concise subquestions that are directly related to the main question. Ensure that each subquestion is clear, straightforward, and aims to retrieve new or complementary information from the RAG system.
            Respond in pure JSON format without any additional text, like this:
            {{
                "sub_questions": [
                    "Subquestion 1",
                    "Subquestion 2",
                    "Subquestion 3"
                ]
            }}
        """
        subquestions_response = ctx.data["llm"].complete(prompt)
        response_obj = json.loads(str(subquestions_response))
        sub_questions = response_obj["sub_questions"]

        ctx.data["question_collect_count"] = len(sub_questions)
        for q in sub_questions:
            ctx.send_event(QueryEvent(query = q))
        return None

    @step(pass_context=True)
    async def execute_query(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:

        ctx.data["question_count"] += 1
        query = QueryBundle(query_str=ev.query)
        response = ctx.data["query_engine"].query(query)
        # print(response)
        return AnswerEvent(
            question = ev.query,
            answer = f"{response!s}",
            source_nodes = response.source_nodes,
        )
    
    @step(pass_context=True)
    async def synthesize(self, ctx: Context, ev: SynthesizeEvent) -> CodingEvent:
        response_synthesizer = get_response_synthesizer()
        final_response = await response_synthesizer.asynthesize(
            query = ev.query,
            nodes = ev.reasoning_nodes,
            additional_source_nodes = ev.source_nodes,
        )
        return CodingEvent(
            question = ev.query,
            answer = f"{final_response!s}",
            source_nodes = ev.source_nodes
        )
    
    @step(pass_context=True)
    async def response_coding(self, ctx: Context, ev: CodingEvent) -> StopEvent:
        contexts = [c.node.get_content() for c in ev.source_nodes]
        final_response = ev.answer
        
        if str(ctx.data["query"]["possible_options"]).lower() != "none":
            prompt = f"""
                Your task is to extract and return only the items from the provided options that appear in the given answer.

                Answer:
                {ev.answer}

                Options:
                {ctx.data["query"]["possible_options"]}

                Requirements:
                Return a comma-separated string of matching items (e.g., "Option1, Option2, ...").
                The output must include only the matching items without any additional text, words, or formatting.
            """
            code = f"{ctx.data["llm"].complete(prompt)!s}"
        else:
            prompt = f"""
                Your task is to condense the provided text into a maximum of one sentence or phrase that directly addresses the given topic.

                Topic: {ctx.data["query"]["topic"]}
                Text: {ev.answer}

                Requirements:
                Return the summary as a plain string without any additional formatting or context.
            """
            code = f"[response not coded] {ctx.data["llm"].complete(prompt)!s}"

        best_contexts = sorted(ev.source_nodes, key=lambda c: c.score, reverse=True)[:5]
        result = {
            "query": ctx.data["query"],
            "question": ctx.data["question"],
            "answer": final_response,
            "code": code,
            "reasoning": [{'question': f'{r['question']}', 'answer': f'{r['answer']}'} for r in ctx.data["reasoning"]],
            "best_context": [{"context": context.get_content().strip(), "score": context.score} for context in best_contexts],
            "contexts": contexts,
        }
        return StopEvent(result = result)

    

    