import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from vertexai import init
from vertexai.generative_models import GenerativeModel, ChatSession
from dotenv import load_dotenv

load_dotenv()

# Initialize Vertex AI once at the top level
# It is a best practice to keep initialization outside of functions
# that might be called multiple times concurrently.
init(project="nodal-vigil-471801-i2", location="us-central1")

def generate_single_response(prompt: str, model_name: str, thread_id: int):
    """
    Generate a single, stateless response using Vertex AI Gemini.
    This is the correct function for one-off parallel requests.
    """
    try:
        model = GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return {
            "thread_id": thread_id,
            "prompt": prompt,
            "response": resp.text,
            "success": True
        }
    except Exception as e:
        return {
            "thread_id": thread_id,
            "prompt": prompt,
            "response": None,
            "error": str(e),
            "success": False
        }

def batch_inference(prompts: list[str], max_workers: int = 5, model_name: str = "gemini-2.5-flash"):
    """
    Run inference on multiple prompts in parallel using threading.
    This function uses generate_single_response for efficiency and correctness.
    """
    print(f"🚀 Starting batch inference with {max_workers} threads...")
    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_prompt = {
            executor.submit(generate_single_response, prompt, model_name, i): (i, prompt)
            for i, prompt in enumerate(prompts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_prompt):
            thread_id, prompt = future_to_prompt[future]
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    print(f"✅ Thread {thread_id}: Completed")
                else:
                    print(f"❌ Thread {thread_id}: Failed - {result['error']}")
            except Exception as e:
                print(f"❌ Thread {thread_id}: Exception - {str(e)}")
                results.append({
                    "thread_id": thread_id,
                    "prompt": prompt,
                    "response": None,
                    "error": str(e),
                    "success": False
                })

    # Sort results by thread_id to maintain order
    results.sort(key=lambda x: x["thread_id"])

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n📊 Batch inference completed in {total_time:.2f} seconds")
    print(f"📈 Average time per request: {total_time/len(prompts):.2f} seconds")
    print(f"⚡ Throughput: {len(prompts)/total_time:.2f} requests/second")

    return results