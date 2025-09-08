# Copyright Sierra

import json
from litellm import completion
from openai import OpenAI
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from tau_bench.model_utils.judge_config import JudgeConfig, DEFAULT_JUDGE_CONFIG
from tau_bench.model_utils.judge_inference import JudgeInference

@dataclass
class CandidateInfo:
    content: str
    mean_logprob: float
    sum_logprob: float
    rank: int
    valid: bool
    num_tokens: int

client = OpenAI(
    base_url="http://localhost:8005/v1",
    api_key="EMPTY"
)

extera_body_qwen3 = {
                    "top_k": 20, 
                    "top_p": 0.8, # 0.95 for thinking = true
                    "min_p": 0,
                    "temperature": 0.7,
                    "chat_template_kwargs": {"enable_thinking": False},
                }

class ChatReActAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        judge_config: JudgeConfig = None,
    ) -> None:
        print("ASS MODEL: ", model)
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        super().__init__(tools_info, wiki, instruction, model, provider, temperature)
        self.use_reasoning = use_reasoning
        self.client = OpenAI(
            base_url="http://localhost:8005/v1",
            api_key="EMPTY"
        )
        
        # Initialize judge inference
        self.judge_config = judge_config or DEFAULT_JUDGE_CONFIG
        self.judge_inference = JudgeInference(
            base_url=self.judge_config.base_url,
            api_key=self.judge_config.api_key,
            model=self.judge_config.model,
            temperature=self.judge_config.temperature,
            max_tokens=self.judge_config.max_tokens
        )

    def solve(self, env: Env) -> SolveResult:
        trajectory = []
        max_steps = 50
        step = 0
        
        while step < max_steps:
            step += 1
            
            # Get current state
            state = env.get_state()
            trajectory.append({"step": step, "state": state})
            
            # Generate action using LLM judge
            action = self._generate_action_with_judge(state, trajectory)
            trajectory.append({"step": step, "action": action})
            
            # Execute action
            result = env.step(action)
            trajectory.append({"step": step, "result": result})
            
            # Check if done
            if result.done:
                break
                
        return SolveResult(trajectory=trajectory, success=result.done)

    def _get_conversation_history(self, trajectory: List[Dict], num_turns: int = 2) -> str:
        """Extract the last num_turns of conversation from trajectory"""
        history = []
        turn_count = 0
        
        # Go backwards through trajectory to find conversation turns
        for entry in reversed(trajectory):
            if "action" in entry and entry["action"].name == RESPOND_ACTION_NAME:
                history.insert(0, f"User: {entry['action'].args.get(RESPOND_ACTION_FIELD_NAME, '')}")
                turn_count += 1
                if turn_count >= num_turns:
                    break
            elif "result" in entry and "response" in entry["result"]:
                history.insert(0, f"Agent: {entry['result']['response']}")
        
        return "\n".join(history)

    def _generate_action_candidates(self, state: str, conversation_history: str) -> List[Dict]:
        """Generate multiple action candidates using the current model"""
        return self.judge_inference.generate_action_candidates(
            state=state,
            conversation_history=conversation_history,
            tools_info=self.tools_info,
            num_candidates=self.judge_config.num_candidates
        )

    def _llm_judge_action_selection(
        self, 
        state: str, 
        conversation_history: str, 
        candidates: List[Dict]
    ) -> Action:
        """Use LLM judge to select the best action from candidates"""
        result = self.judge_inference.select_best_action(
            state=state,
            conversation_history=conversation_history,
            candidates=candidates,
            tools_info=self.tools_info
        )
        
        return Action(
            name=result["selected_candidate"]["action_name"],
            args=result["selected_candidate"]["action_args"]
        )

    def _generate_action_with_judge(self, state: str, trajectory: List[Dict]) -> Action:
        """Generate action using LLM judge instead of log probability ranking"""
        # Get conversation history (last 2 turns)
        conversation_history = self._get_conversation_history(trajectory, self.judge_config.conversation_turns)
        
        # Generate multiple action candidates
        candidates = self._generate_action_candidates(state, conversation_history)
        
        # Use LLM judge to select best action
        best_action = self._llm_judge_action_selection(state, conversation_history, candidates)
        
        return best_action

    # Keep the original method for backward compatibility
    def _generate_action(self, state: str, trajectory: List[Dict]) -> Action:
        """Original action generation method - now uses judge system"""
        return self._generate_action_with_judge(state, trajectory)

        
REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""



ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""
