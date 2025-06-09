
import base64
from openai import OpenAI
import os
import scipdf
from google import genai
from google.genai import types
import re
import shutil


key = ""
gemini_api_key = ""
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_figures_and_tables_from_papers(output,full_path):
    shutil.rmtree(output, ignore_errors=True)
    os.makedirs(output, exist_ok=True)

    if os.path.isfile(full_path):
        scipdf.parse_figures(full_path, output_folder=output)
        print(full_path + " finished")

def analyze_figures_and_tables(folder):
    client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
            )
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        if os.path.isfile(full_path):
            image_path = full_path
            base64_image = encode_image_to_base64(image_path)
            data_url = f"data:image/png;base64,{base64_image}"

            completion = client.chat.completions.create(
            extra_body={},
            model="google/gemma-3-27b-it:free",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "What is in this image?"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                                "url": data_url
                            }
                    }
                ]
                }
            ]
            )
            print(completion.choices[0].message.content)
            print(full_path + " finished")

def analyze_figures_and_tables_with_gemma(folder):
    client = genai.Client(api_key=gemini_api_key)
    responses = []
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        if os.path.isfile(full_path):
            with open(full_path, 'rb') as f:
                image_bytes = f.read()

            response = client.models.generate_content(
                model='gemma-3-27b-it',
                contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                'Describe the contents of this image'
                ]
            )
            
            print(response.text)
            responses.append(response.text)
            # name = re.sub('.png','',filename)
            # with open(os.path.join(output_folder,name)+".txt",'w') as f:
            #     f.write(response.text)
    return responses
# get_figures_and_tables_from_papers("outputs","input/12550.pdf")
# l = analyze_figures_and_tables_with_gemma("outputs/figures")
# print("aaa")