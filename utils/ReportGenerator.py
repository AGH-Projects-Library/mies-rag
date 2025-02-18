import os
import csv
import json

from config.config import (
    INPUT_PATH, 
    OUTPUT_PATH, 
    STORAGE_PATH, 
    API,
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

class ReportGenerator:
    def __init__(self, queries, output_dir):
        self.queries = queries
        self.output_dir = output_dir
        self.files_results = []

    def generate_partial_report(self, file, info, results):
        self.files_results.append({
            "file_name": file,
            "info": info,
            "result": results
        })
        os.makedirs(f"{self.output_dir}/{file}", exist_ok=True)
        file_path = f"{self.output_dir}/{file}/{file}_result"
        content = ""
        
        # markdown
        for i, result in enumerate(results):
            content += f"## Question {i+1}\n"
            content += f"```json\n{json.dumps(result, indent=4)}\n```\n"
            content += "___\n"
        with open(f"{file_path}.md", "w", encoding="utf-8") as f:
            f.write(content)

        # json
        data = {
            'results': results,
        }
        with open(f"{file_path}.json", 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        return 

    def generate_main_report(self):
        # todo add info (author, title, etc.)
        header_line_1 = ["file"]+[item for q in self.queries for item in (q["topic"], "", "")]
        header_line_2 = [""]+[item for q in self.queries for item in ("[context = original text]", "[LLM answer]", f"[kod]:\n{q['possible_options']}")]

        data_rows = []
        for entry in self.files_results:
            data_row = [entry["file_name"]]
            for r in entry["result"]:
                context = ""
                for i, c in enumerate(r["best_context"]):
                    context += f"{i+1}.  {c["context"]} \n"
                data_row.append(context)
                data_row.append(r["answer"])
                data_row.append(r["code"])
            data_rows.append(data_row)

        with open(f"{self.output_dir}/raport.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            writer.writerow(header_line_1)
            writer.writerow(header_line_2)

            writer.writerows(data_rows)

        return
    
    def generate_config_report(self, execution_time):

        obj = {
            "config": {
                "api": API,
                "model": MODEL,
                "input_path": INPUT_PATH,
                "output_path": OUTPUT_PATH,
                "storage_path": STORAGE_PATH,
                "max_steps": MAX_STEPS,
                "disable_second_loop": DESABLE_SECOND_LOOP,
                "evaluation": EVALUATION,
                "ragas": RAGAS,
                "g_eval": G_EVAL,
                "ground_truth": GROUND_TRUTH,
                "clear_storage": CLEAR_STORAGE,
                "cohere_rerank": COHERE_RERANK,
            },
            "queries": self.queries,
            "files": [r["file_name"] for r in self.files_results],
            "execution_time": execution_time
        }

        with open(f"{self.output_dir}/start_up_detail.json", 'w', encoding='utf-8') as file:
            json.dump(obj, file, ensure_ascii=False, indent=4)
      
