from gemini_config import generate_single_response, batch_inference

def single_inference_example():
    """Example of single inference"""
    print("🔹 Single Inference Example:")
    prompt = "Give me a two-line summary of the theory of relativity."
    result = generate_single_response(prompt)
    if result["success"]:
        print(f"Prompt: {prompt}")
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result['error']}")

def batch_inference_example():
    """Example of batch inference with threading"""
    print("\n🔹 Batch Inference Example:")
    
    # Sample prompts for testing
    prompts = [
        "Explain quantum computing in one sentence.",
        "What is the capital of Japan?",
        "Describe machine learning briefly.",
        "What is photosynthesis?",
        "Explain gravity in simple terms.",
        "What is the speed of light?",
        "Describe artificial intelligence.",
        "What is DNA?",
        "Explain the water cycle.",
        "What is the theory of evolution?"
    ]
    
    # Run batch inference with 3 threads (adjust as needed)
    results = batch_inference(prompts, max_workers=3)
    
    # Display results
    print("\n📋 Results:")
    for i, result in enumerate(results):
        if result["success"]:
            print(f"{i+1}. ✅ {result['response']}")
        else:
            print(f"{i+1}. ❌ Error: {result['error']}")

if __name__ == "__main__":
    # Run examples
    single_inference_example()
    batch_inference_example()
