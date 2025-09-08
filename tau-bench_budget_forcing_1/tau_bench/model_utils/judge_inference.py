# tau_bench/model_utils/judge_inference.py
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json

class JudgeInference:
    """Handles LLM-based action selection for agents"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:3001/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen2.5-72B-Instruct",
        temperature: float = 0.1,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def select_best_action(
        self,
        state: str,
        conversation_history: str,
        candidates: List[Dict],
        tools_info: List[Dict[str, Any]]
    ) -> Dict:
        """Select the best action from candidates using LLM judgment"""
        
        candidates_text = "\n".join([
            f"{i+1}. {candidate['action_name']}({candidate['action_args']}) - {candidate['reasoning']}"
            for i, candidate in enumerate(candidates)
        ])
        
        judge_prompt = f"""
You are an expert judge for selecting the best action in a conversational AI agent.

Current state: {state}

Conversation history:
{conversation_history}

Available tools: {json.dumps(tools_info, indent=2)}

Action candidates:
{candidates_text}

Please select the BEST action by considering:
1. Relevance to the current conversation context
2. Likelihood of making progress toward the user's goal
3. Appropriateness of the action given the current state
4. Logical flow from the conversation history

Respond with ONLY the number (1-{len(candidates)}) of the best action, followed by a brief explanation.

Example: "3. This action directly addresses the user's request and uses the most appropriate tool."
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        judge_response = response.choices[0].message.content.strip()
        
        # Parse the selected action number
        try:
            selected_num = int(judge_response.split('.')[0]) - 1
            if 0 <= selected_num < len(candidates):
                return {
                    "selected_candidate": candidates[selected_num],
                    "judge_reasoning": judge_response,
                    "success": True
                }
        except (ValueError, IndexError):
            pass
        
        # Fallback: return first candidate
        return {
            "selected_candidate": candidates[0],
            "judge_reasoning": "Fallback to first candidate",
            "success": False
        }
    
    def generate_action_candidates(
        self,
        state: str,
        conversation_history: str,
        tools_info: List[Dict[str, Any]],
        num_candidates: int = 5
    ) -> List[Dict]:
        """Generate multiple action candidates for the judge to evaluate"""
        
        prompt = f"""
You are an AI agent that needs to take an action. Generate {num_candidates} different possible actions.

Current state: {state}

Conversation history:
{conversation_history}

Available tools: {json.dumps(tools_info, indent=2)}

For each action, provide:
1. The action name
2. The action arguments
3. A brief reasoning for why this action might be good

Format as JSON:
{{
    "candidates": [
        {{
            "action_name": "tool_name",
            "action_args": {{"arg1": "value1"}},
            "reasoning": "Why this action might be good"
        }},
        ...
    ]
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temperature for diversity
            max_tokens=500
        )
        
        try:
            candidates_data = json.loads(response.choices[0].message.content)
            return candidates_data.get("candidates", [])
        except json.JSONDecodeError:
            # Fallback: generate single candidate
            return [{"action_name": "think", "action_args": {}, "reasoning": "Default action"}]