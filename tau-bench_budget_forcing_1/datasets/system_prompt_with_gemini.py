from typing import List, Dict, Any
import json
from tqdm import tqdm
import os
import time
from gemini_config import generate_single_response

def load_json(path_file):
    with open(path_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(data, path_file):
    with open(path_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4) 

def create_prompt(system_prompt: str, user_prompt: str, human_messages: list) -> str:
    """
    Create a single prompt by combining system prompt, user prompt, and conversation data.
    """
    current_user_prompt = user_prompt + f"{json.dumps(human_messages)}\n"
    full_prompt = f"{system_prompt}\n\n{current_user_prompt}"
    return full_prompt

from datasets import load_dataset

dataset = load_dataset("Salesforce/APIGen-MT-5k")
print(len(dataset["train"]))

system_prompt = """You are an AI assistant that reconstructs user simulation instructions from conversations.
You will be given a conversation where the "user" is a simulated persona following a hidden instruction.
Your task is to infer and write the original user instruction as completely and precisely as possible.

Output Format:
Return only one paragraph with the instruction, without any extra explanation.
"""

user_prompt = """
Example:
Conversation:
[{'from': 'human', 'value': 'Hello, I have a request regarding a past flight reservation, 0U4NPP, from PHL to DEN. I was wondering if I could still cancel it due to personal reasons.'}, {'from': 'human', 'value': "Sure. My user ID is amelia_rossi_1651. The reason for the cancellation request is personal. I understand if it's an unusual request as the flights have already landed, but I thought it was worth checking if there's any flexibility."} {'from': 'human', 'value': 'I understand and appreciate your help. No need to transfer me to a human agent. Have a good day!'}]

User Instruction:
Your user id is amelia_rossi_1651. You previously booked a flight reservation with confirmation code 0U4NPP, from Philadelphia (PHL) to Denver (DEN). You want to ask if it is still possible to cancel the flight due to personal reasons, even though the flights have already landed. You understand this is an unusual request, but you would like to check if there is any flexibility. If offered, you do not want to be transferred to a human agent. You will remain polite, appreciative, and reactive to the agent's guidance.

Conversation:
"""

output_path = "datasets/api-gen-mt-user"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

new_res = []

# Process all conversations sequentially
for i in tqdm(range(len(dataset["train"]))):
    conversation = dataset["train"][i]["conversations"]
    human_messages = [item["value"] for item in conversation if item["from"] == "human"]
    
    # Create the full prompt for this conversation
    full_prompt = create_prompt(system_prompt, user_prompt, human_messages)
    
    try:
        # Generate response using single inference
        result = generate_single_response(full_prompt, "gemini-2.5-flash", i)
        
        if result["success"]:
            response_text = result["response"]
        else:
            response_text = None
            print(f"❌ Item {i}: Failed - {result.get('error', 'Unknown error')}")
        
        tmp_dict = {
            "id": i,
            "conversation": conversation,
            "user_instruction": response_text
        }
        
        new_res.append(tmp_dict)
        
        # Save progress every 100 items
        if (i + 1) % 100 == 0:
            with open(os.path.join(output_path, "api_gen_reconstruction_gemini_progress.json"), "w") as f:
                json.dump(new_res, f, indent=4)
            print(f"💾 Progress saved at item {i + 1}")
        
        # Small delay to be respectful to the API
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Error processing item {i}: {e}")
        # Save failed item info
        failed_item = {
            "id": i,
            "conversation": conversation,
            "error": str(e)
        }
        new_res.append(failed_item)

# Save final results
with open(os.path.join(output_path, "api_gen_reconstruction_gemini_final.json"), "w") as f:
    json.dump(new_res, f, indent=4)

print(f"✅ Processing complete! Results saved to {output_path}/")
print(f"📊 Total items processed: {len(new_res)}")
