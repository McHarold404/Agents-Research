"""
Extract action sequences from APIGen-MT-5k dataset.
This script parses conversations to extract the sequence of function calls made by the assistant.
"""

import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import defaultdict, Counter

def load_json(path_file: str) -> Any:
    """Load JSON data from file"""
    with open(path_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(data: Any, path_file: str) -> None:
    """Save data to JSON file"""
    with open(path_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def parse_function_call(value: str) -> Optional[Dict[str, Any]]:
    """Parse function call JSON and extract name and arguments"""
    try:
        parsed = json.loads(value)
        return {
            "name": parsed.get("name"),
            "arguments": parsed.get("arguments", {})
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ Failed to parse function call: {value[:100]}... Error: {e}")
        return None

def extract_action_sequence(conversation: List[Dict[str, Any]], conversation_id: int) -> Dict[str, Any]:
    """Extract action sequence from a single conversation"""
    
    action_sequence = []
    step = 1
    
    # Get conversation context (first human message)
    conversation_context = None
    for msg in conversation:
        if msg.get("from") == "human":
            conversation_context = msg.get("value", "")
            break
    
    # Process conversation messages
    for i, message in enumerate(conversation):
        if message.get("from") == "function_call":
            # Parse the function call
            function_data = parse_function_call(message.get("value", ""))
            if not function_data:
                continue
            
            # Get following observation (result)
            action_output = None
            for j in range(i+1, len(conversation)):
                if conversation[j].get("from") == "observation":
                    action_output = conversation[j].get("value", "")
                    break
            
            # Create action entry
            action = {
                "step": step,
                "function_name": function_data["name"],
                "arguments": function_data["arguments"],
                "action_output": action_output
            }
            
            action_sequence.append(action)
            step += 1
    
    return {
        "conversation_id": conversation_id,
        "conversation_context": conversation_context,
        "action_sequence": action_sequence,
        "total_actions": len(action_sequence),
        "has_actions": len(action_sequence) > 0
    }

def analyze_function_usage(action_sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze function usage patterns across all conversations"""
    
    function_counts = Counter()
    function_sequences = defaultdict(list)
    conversation_lengths = []
    action_patterns = defaultdict(int)
    
    for conv_data in action_sequences:
        if not conv_data["has_actions"]:
            continue
        
        conversation_lengths.append(conv_data["total_actions"])
        
        # Count individual functions
        for action in conv_data["action_sequence"]:
            function_counts[action["function_name"]] += 1
        
        # Track function sequences
        if conv_data["total_actions"] > 1:
            sequence = [action["function_name"] for action in conv_data["action_sequence"]]
            function_sequences[conv_data["total_actions"]].append(sequence)
            
            # Track common patterns (pairs)
            for i in range(len(sequence) - 1):
                pattern = f"{sequence[i]} -> {sequence[i+1]}"
                action_patterns[pattern] += 1
    
    return {
        "function_frequency": dict(function_counts.most_common()),
        "conversation_length_stats": {
            "total_conversations": len(action_sequences),
            "conversations_with_actions": sum(1 for c in action_sequences if c["has_actions"]),
            "average_actions": sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0,
            "max_actions": max(conversation_lengths) if conversation_lengths else 0,
            "min_actions": min(conversation_lengths) if conversation_lengths else 0
        },
        "common_action_patterns": dict(Counter(action_patterns).most_common(20)),
        "function_sequences_by_length": {
            str(length): sequences[:10] for length, sequences in function_sequences.items()
        }
    }

def main():
    """Main function to extract action sequences from APIGen-MT-5k dataset"""
    
    print("🚀 Starting action sequence extraction from APIGen-MT-5k dataset...")
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset = load_json("APIGen-MT-5k_train.json")
    print(f"✅ Loaded {len(dataset)} conversations")
    
    # Create output directory
    output_dir = "action_sequences"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract action sequences
    print("🔍 Extracting action sequences...")
    action_sequences = []
    
    for i, item in enumerate(tqdm(dataset, desc="Processing conversations")):
        conversation = item.get("conversations", [])
        sequence_data = extract_action_sequence(conversation, i)
        action_sequences.append(sequence_data)
        
        # Save progress every 1000 items
        if (i + 1) % 1000 == 0:
            progress_file = os.path.join(output_dir, f"action_sequences_progress_{i+1}.json")
            save_json(action_sequences, progress_file)
            print(f"💾 Progress saved at conversation {i + 1}")
    
    # Save complete results
    print("💾 Saving complete results...")
    save_json(action_sequences, os.path.join(output_dir, "action_sequences_full.json"))
    
    # Generate analysis
    print("📊 Generating analysis...")
    analysis = analyze_function_usage(action_sequences)
    save_json(analysis, os.path.join(output_dir, "action_sequences_analysis.json"))
    
    # Create summary
    summary = {
        "dataset_info": {
            "total_conversations": len(dataset),
            "conversations_with_actions": sum(1 for c in action_sequences if c["has_actions"]),
            "conversations_without_actions": sum(1 for c in action_sequences if not c["has_actions"])
        },
        "function_summary": analysis["function_frequency"],
        "conversation_stats": analysis["conversation_length_stats"]
    }
    save_json(summary, os.path.join(output_dir, "action_sequences_summary.json"))
    
    # Print results
    print("\n📈 Extraction Results:")
    print(f"✅ Total conversations processed: {len(dataset)}")
    print(f"✅ Conversations with actions: {summary['dataset_info']['conversations_with_actions']}")
    print(f"✅ Conversations without actions: {summary['dataset_info']['conversations_without_actions']}")
    print(f"✅ Average actions per conversation: {analysis['conversation_length_stats']['average_actions']:.2f}")
    print(f"✅ Most used function: {list(analysis['function_frequency'].keys())[0] if analysis['function_frequency'] else 'None'}")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("   - action_sequences_full.json (complete data)")
    print("   - action_sequences_analysis.json (detailed analysis)")
    print("   - action_sequences_summary.json (summary statistics)")

if __name__ == "__main__":
    main()
