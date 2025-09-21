
from typing import List, Dict, Any
from openai import OpenAI
import json
from tqdm import tqdm
import os
import openai 
import re
from openai import OpenAI
from tqdm import tqdm
import time


MODEL = "Qwen/Qwen2.5-72B-Instruct"
# client = OpenAI()
client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY"
)

def load_json(path_file):
    with open(path_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(data, path_file):
    with open(path_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4) 

# def run_agent(system_prompt: str, user_prompt: str) -> str:
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#     resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
#     return resp.choices[0].message.content

def run_agent(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 20,
    retry_delay: int = 3 
) -> str:
    """
    Run the agent with retry mechanism for token limit errors.

    Args:
        system_prompt (str): System role prompt
        user_prompt (str): User role prompt
        max_retries (int): Number of retries if token error happens
        retry_delay (int): Delay before retrying (seconds)

    Returns:
        str: Model's response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2
            )
            return resp.choices[0].message.content

        except Exception as e:
            if "max tokens" in str(e).lower() or "400" in str(e):
                if attempt < max_retries:
                    print(f"⚠️ Attempt {attempt} failed due to token limit. Waiting {retry_delay//60} minutes before retry...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"❌ Max retries ({max_retries}) reached. Last error: {e}")
            else:
                raise e

from datasets import load_dataset

dataset = load_dataset("Salesforce/APIGen-MT-5k", cache_dir="/scr/amir/tool_calling_datasets")
print(len(dataset["train"]))

system_prompt = """You are an AI assistant that reconstructs user simulation instructions from conversations.
You will be given a conversation where the “user” is a simulated persona following a hidden instruction.
Your task is to infer and write the original user instruction as completely and precisely as possible.

Output Format:
Return only one paragraph with the instruction, without any extra explanation.
"""

user_prompt = """
Example:
Conversation:
[{'from': 'human', 'value': 'Hello, I have a request regarding a past flight reservation, 0U4NPP, from PHL to DEN. I was wondering if I could still cancel it due to personal reasons.'}, {'from': 'human', 'value': "Sure. My user ID is amelia_rossi_1651. The reason for the cancellation request is personal. I understand if it's an unusual request as the flights have already landed, but I thought it was worth checking if there's any flexibility."} {'from': 'human', 'value': 'I understand and appreciate your help. No need to transfer me to a human agent. Have a good day!'}]

User Instruction:
Your user id is amelia_rossi_1651. You previously booked a flight reservation with confirmation code 0U4NPP, from Philadelphia (PHL) to Denver (DEN). You want to ask if it is still possible to cancel the flight due to personal reasons, even though the flights have already landed. You understand this is an unusual request, but you would like to check if there is any flexibility. If offered, you do not want to be transferred to a human agent. You will remain polite, appreciative, and reactive to the agent’s guidance.

Conversation:
"""

output_path = "/scr/amir/api_gen_reconstruction"

new_res = []

# path_id_failed_file = "/scr/amir/api_gen_reconstruction/failed_ids_4.json"

failed_id = load_json(path_id_failed_file)

# for i in tqdm(failed_id[:1500]):
for i in tqdm(range(len(dataset["train"]))):
    conversation = dataset["train"][i]["conversations"]
    human_messages = [item["value"] for item in conversation if item["from"] == "human"]
    # print(human_messages)
    user_prompt += f"{json.dumps(human_messages)}\n"
    response = run_agent(system_prompt, user_prompt)
    time.sleep(5)

    tmp_dict = {
        "id": i,
        "conversation": conversation,
        "user_instruction": response
    }

    new_res.append(tmp_dict)

    with open(os.path.join(output_path, "api_get_reconstrcution_7.json"), "w") as f:
        json.dump(new_res, f, indent=4)
