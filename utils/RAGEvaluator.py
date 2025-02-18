import json
import os
from langchain_groq import ChatGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate as ragasEvaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    SemanticSimilarity,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
    AnswerCorrectness,
)
from ragas import SingleTurnSample
from ragas import EvaluationDataset as RagasEvaluationDataset

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase


class RAGEvaluator:
    def __init__(self, start_questions, llm, results, model, evaluate_ragas, evaluate_geval, ground_truth, filename):
        self.start_questions = start_questions
        self.llm = llm
        self.results = results
        self.model = model
        self.evaluate_ragas = evaluate_ragas
        self.evaluate_geval = evaluate_geval
        self.ground_truth = ground_truth
        self.filename = filename
        self.samples = self.create_samples()
        
    def create_samples(self):
        ragas_samples = []
        geval_samples = []
        for i, r in enumerate(self.results):
            if str(r["query"]["possible_options"]).lower() != "none":
                a = r["code"]
            else:
                a = r["answer"]
            ragas_sample = SingleTurnSample(
                user_input = r["query"]["topic"],
                retrieved_contexts = r["contexts"],
                response = a,
                reference = self.ground_truth[i],
            )
            ragas_samples.append(ragas_sample)
            geval_sample = LLMTestCase(
                input = r["query"]["topic"],
                actual_output = a,
                expected_output = self.ground_truth[i],
                retrieval_context = r["contexts"],
            )
            geval_samples.append(geval_sample)
        return {"ragas": ragas_samples, "geval": geval_samples}

    def RAGAS(self):
        print("\nRAGAS evaluation")
        dataset = RagasEvaluationDataset(samples=self.samples["ragas"])
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=self.model))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        metrics = [
            LLMContextRecall(llm=evaluator_llm), 
            Faithfulness(llm=evaluator_llm),
            SemanticSimilarity(embeddings=evaluator_embeddings),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
            
        ]
        result = ragasEvaluate(dataset=dataset, metrics=metrics)

        dataframe = result.to_pandas()
        columns_to_include = dataframe.columns[4:] 
        result = [
            {col: row[col] for col in columns_to_include}
            for _, row in dataframe.iterrows()
        ]
        obj = {"samples": result}
        with open(f"ragas_result.json", 'w', encoding='utf-8') as file:
            json.dump(obj, file, ensure_ascii=False, indent=4)
        return result

    def DEEPEVAL(self):
        print("\nDEEPEVAL evaluation")
        metrics = [
            ContextualPrecisionMetric(threshold=0.5, model=self.model, include_reason=False, verbose_mode=False),
            ContextualRecallMetric(threshold=0.5, model=self.model, include_reason=False, verbose_mode=False),
            ContextualRelevancyMetric(threshold=0.5, model=self.model, include_reason=False, verbose_mode=False),
            AnswerRelevancyMetric(threshold=0.5, model=self.model, include_reason=False, verbose_mode=False),
            FaithfulnessMetric(threshold=0.5, model=self.model, include_reason=False, verbose_mode=False),
            ]
        dataset = EvaluationDataset(test_cases=self.samples["geval"])
        dataset_result = dataset.evaluate(metrics) 
        result = []
        for test_result in dataset_result.test_results:
            scores = {}
            for metric_data in test_result.metrics_data:
                scores[metric_data.name.lower().replace(" ", "_")] = metric_data.score
            result.append(scores)
        return result

    def evaluate(self):
        geval_result = self.DEEPEVAL() if self.evaluate_geval else [None] * len(self.results)
        ragas_result = self.RAGAS() if self.evaluate_ragas else [None] * len(self.results)
        return [{"ragas": ragas_result[i],"geval": geval_result[i]} for i in range(len(self.results))]
