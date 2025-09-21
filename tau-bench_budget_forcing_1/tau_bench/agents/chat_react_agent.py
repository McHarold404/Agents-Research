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
    ) -> None:
        print("ASS MODEL: ", model)
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        # print(tools_info[0].keys())

        tools_info = [tool for tool in tools_info if "think" not in tool["function"]["name"]]

        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        # self.model = model
        self.model = f"Qwen/{model}"
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        # res = completion(
        #     model=self.model,
        #     custom_llm_provider=self.provider,
        #     messages=messages,
        #     temperature=self.temperature,
        # )

        force_time = 1
        message, candidates_info = self.sample_best_of_N_Actions(messages, force_time)
        #input()

        # message = res.choices[0].message
        # print(message.content)
        action_str = message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        
        # print(action)
        # print(message.model_dump())
        # input("Next?")

        # Add candidate information to the message
        message_dict = message.model_dump()
        message_dict['candidates'] = [candidate.__dict__ for candidate in candidates_info]
        return message_dict, action, 0
    
    ## TODO: Update this to use Best of N (N = 3)
    ## store log probs for all and choose best answer. 
    ## 
    def sample_best_of_N_Actions(self, messages, force_time) -> Tuple[Dict[str, Any], List[CandidateInfo]]:
        """
        Minimal change: this function now does Best-of-N sampling and returns the chosen message
        along with detailed information about all candidates including their log probabilities and ranks.
        It scores candidates by mean logprob over the JSON that follows the final 'Action:' tag.
        """
        # ---- local helpers (kept inside to avoid changing class surface) ----
        def _join_tokens_and_spans(tokens):
            out, spans, pos = [], [], 0
            for t in tokens:
                s = t.token
                out.append(s)
                start, end = pos, pos + len(s)
                spans.append((start, end))
                pos = end
            return "".join(out), spans

        def _find_action_json_span(text: str) -> Optional[Tuple[int, int]]:
            anchor = text.rfind("Action:")
            if anchor == -1:
                return None
            i = text.find("{", anchor)
            if i == -1:
                return None
            depth = 0
            for j in range(i, len(text)):
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return (i, j + 1)
            return None

        def _mean_logprob_on_span(choice) -> float:
            """
            Scores by mean logprob over the Action JSON.
            If Action.name == RESPOND_ACTION_NAME, instead score only the
            arguments[RESPOND_ACTION_FIELD_NAME] content string.
            """
            toks = choice.logprobs.content or []
            full_text, spans = _join_tokens_and_spans(toks)

            # 1) Find Action JSON span
            span = _find_action_json_span(full_text)
            if span is None:
                return float("-inf")
            start_char, end_char = span

            # tokens fully inside the Action JSON
            idxs = [i for i, (s, e) in enumerate(spans) if s >= start_char and e <= end_char]
            if not idxs:
                return float("-inf")

            # default: mean over entire Action JSON
            default_vals = [toks[i].logprob for i in idxs]
            default_mean = float(sum(default_vals)) / len(default_vals)

            # 2) If RESPOND action, try to score only the content string
            action_text = full_text[start_char:end_char]
            try:
                payload = json.loads(action_text)
            except Exception:
                return default_mean  # fall back if malformed

            if isinstance(payload, dict) and payload.get("name") == RESPOND_ACTION_NAME:
                args = payload.get("arguments", {})
                if isinstance(args, dict) and RESPOND_ACTION_FIELD_NAME in args:
                    content_value = args[RESPOND_ACTION_FIELD_NAME]
                    if isinstance(content_value, str):
                        # Try a few needle encodings to locate the emitted content in the JSON text
                        needles = [
                            json.dumps(content_value, ensure_ascii=False),  # canonical with quotes/escapes
                            json.dumps(content_value),                     # ASCII-escaped variant
                            f"\"{content_value}\"",                         # raw quoted
                            content_value,                                  # raw (last resort)
                        ]
                        for needle in needles:
                            rel = action_text.find(needle)
                            if rel != -1:
                                c_start = start_char + rel
                                c_end = c_start + len(needle)
                                c_idxs = [i for i, (s, e) in enumerate(spans) if s >= c_start and e <= c_end]
                                if c_idxs:
                                    c_vals = [toks[i].logprob for i in c_idxs]
                                    return float(sum(c_vals)) / len(c_vals)

            # If we couldn't isolate the content span, use the full-JSON mean
            return default_mean

        # ---- Best-of-N sampling (single hop; no iterative rethink) ----
        BEST_OF = 5  # adjust if needed
        res = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.6,            # keep your current setup
            n=BEST_OF,
            logprobs=True,              # correct key (vs log_probs)
            top_logprobs=0,             # set >0 if you want alternative tokens returned too
            extra_body=extera_body_qwen3
        )

        # score each candidate by mean logprob over Action JSON
        scores = [(_mean_logprob_on_span(ch), idx) for idx, ch in enumerate(res.choices)]
        scores.sort(key=lambda x: x[0], reverse=True)

        # Create simplified candidate information
        candidates_info = []
        for rank, (score, idx) in enumerate(scores):
            choice = res.choices[idx]
            toks = choice.logprobs.content or []

            # Calculate sum logprob (use all tokens for simplicity)
            sum_logprob = score
            if toks:
                logprob_values = [t.logprob for t in toks]
                sum_logprob = float(sum(logprob_values))

            candidate = CandidateInfo(
                content=choice.message.content,
                mean_logprob=score,
                sum_logprob=sum_logprob,
                rank=rank + 1,
                valid=score != float("-inf"),
                num_tokens=len(toks)
            )
            candidates_info.append(candidate)

            # DEBUGGING
            # print("**********************************")
            # print(f"Rank {rank + 1}: Log prob {score}, Valid: {score != float('-inf')}")
            # print(f"Content: {choice.message.content[:100]}...")
            # print("**********************************")

        best_idx = scores[0][1]
        return res.choices[best_idx].message, candidates_info

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        
        # print(response.observation)
        total_cost = 0.0
        info = {}
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            # print(messages[1:])
            # print(obs)
            # input("wait")
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost
            if response.done:
                break
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )
        
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
