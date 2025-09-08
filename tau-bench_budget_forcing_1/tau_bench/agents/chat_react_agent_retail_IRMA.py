# Copyright Sierra

import json
from litellm import completion
from tau_bench.agents.base import Agent
from openai import OpenAI
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple
import os


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
irma_client = OpenAI(
    base_url="http://localhost:8003/v1",
    api_key="EMPTY"
)

extera_body_qwen3 = {
                    "top_k": 20, 
                    "top_p": 0.8,
                    "min_p": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }

class ChatFACTAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
    ) -> None:
        # instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        instruction = FOLLOW_UP_QUESTION_INSTRUCTION
        cons_prompt = CONSTRAINT_PROMPT

        # assistant_wiki = wiki.split("Book flight")[0].strip() if "Airline" in wiki else wiki.split("Cancel pending order")[0].strip()
        self.prompt = (
             wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.wiki = wiki
        self.domain_constraints = "Book flight\n" + wiki.split("Book flight")[-1].strip() if "Airline" in wiki else "Cancel pending order\n" + wiki.split("Cancel pending order")[-1].strip()
        self.model = f"microsoft/{model}"
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
        res = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            #extra_body=extera_body_qwen3
        )

        message = res.choices[0].message
        agent_full_response = message.content
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
        return message.model_dump(), action, 0, agent_full_response

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]

        
        total_cost = 0.0
        info = {}
        previous_user_query = []
        try_flag = True
        for _ in range(max_num_steps):


            if messages[-1]["role"] == "user" and "API output:" not in messages[-1]["content"]:
                user_query = response.observation
                previous_user_query.append(user_query)
                memory_prompt = "<memory>\nUser Query: " + "\nUser Query: ".join(previous_user_query) + "\n</memory>"
                selected_tools = "<tool_suggested>\n" + self.tool_selector_agent(user_query, json.dumps(self.tools_info)) + "\n</tool_suggested>"

                domain_extracted = self.domain_extractor_agent(user_query, self.domain_constraints)
                messages[-1]["content"] = memory_prompt + "\n" + domain_extracted + "\n" + selected_tools

            
            # print(messages[-1]["content"])
            # input("Continue? ")

            message, action, cost, agent_full_response = self.generate_next_step(messages)
            # print(agent_full_response)
            # input("Continue? ")
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                # conversation = messages[1:]
                # user_query = user_query
                tool_call = agent_full_response
                tool_output = obs

                # violation_msg = self.violation_finder_agent(self.domain_constraints, user_query, tool_call)                
                # fb_out = self.feedback_generator_agent(self.domain_constraints, user_query, tool_call, tool_output)
                # push_constraint = f"\n\n{fb_out}. First solve a request and then go throught with another request."
                # obs = "API output: " + obs + push_constraint

                general_domain_constrains = self.domain_extractor_agent(memory_prompt, self.domain_constraints)
                obs = "API output: " + obs + f"\n{general_domain_constrains}"

                # mm_p = "User Query: " + "\nUser Query: ".join(previous_user_query[-2:])

                # signal_constriants = self.check_list_generator_agent(mm_p, self.domain_constraints)

                # obs = "API output: " + obs + f"\n{general_domain_constrains}\n{signal_constriants}"

                # print(mm_p)
                # print(signal_constriants)
                # input("Continue? ")

                #\n\n <feedback>\nDoes the tool call and its output follow the domain policy? If not ask a relevent question from user and try solve the problem.\n</feedback>"
                # print(obs)
                # input("Continue? ")


            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost
            # print(messages[1:])
            # input("Continue? ")
            if response.done:
                break
            
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )
    
    def domain_extractor_agent(self, user_query, domain_constraints):

        sp_domain_extractor_agent = SP_DX_AGNET
        domain_extractor_prompt = DX_PROMPT.format(CONS=domain_constraints, USER_QUERY=user_query)

        messages = [
            {"role": "system", "content": sp_domain_extractor_agent},
            {"role": "user", "content": domain_extractor_prompt},
        ]
        # res = completion(
        #     model=self.model,
        #     custom_llm_provider=self.provider,
        #     messages=messages,
        #     temperature=self.temperature,
        # )
        res = irma_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=2048
            #extra_body=extera_body_qwen3
        )

        message = res.choices[0].message

        return message.model_dump()["content"]
    
    # Not used in the current implementation
    def violation_finder_agent(self, domain_constraints, user_query, tool_call):
        system_prompt = SP_VIOLATION_AGENT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Domain policies:{domain_constraints}\n\nUser Query: {user_query}\n\nAction of agent: {tool_call}"},
        ]
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )

        message = res.choices[0].message

        return message.model_dump()["content"]

    def tool_selector_agent(self, user_query, tool_list):
        system_prompt_tool_selection = SP_TOOL_SELECTION
        messages = [
            {"role": "system", "content": system_prompt_tool_selection},
            {"role": "user", "content": f"Available Tools:{tool_list}\n\nUser Query: {user_query}"},
        ]
        # res = completion(
        #     model=self.model,
        #     custom_llm_provider=self.provider,
        #     messages=messages,
        #     temperature=self.temperature,
        # )
        res = irma_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=2048,
            #extra_body=extera_body_qwen3
        )

        message = res.choices[0].message

        return message.model_dump()["content"]

    # Not used in the current implementation
    def feedback_generator_agent(self, domain_constraints, user_query, tool_call, tool_output):
        
        sp_feedback_generator_agent = SP_FG_AGENT
        fg_prompt = FG_PROMPT.format(USER_QUERY=user_query, TOOL_CALL=tool_call, TOOL_RESPONSE=tool_output, DOMAIN_CONS=domain_constraints)

        messages = [
            {"role": "system", "content": sp_feedback_generator_agent},
            {"role": "user", "content": fg_prompt},
        ]
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )

        message = res.choices[0].message

        return message.model_dump()["content"]
    
    # Not used in the current implementation
    def check_list_generator_agent(self, memory, domain_constraints):
        
        sp_check_list_agent = SP_CHECK_LIST_AGENT

        cl_prompt = CHECK_LIST_AGENT_PROMPT.format(DOMAIN_POLICY=domain_constraints, USER_MEMORY=memory)

        messages = [
            {"role": "system", "content": sp_check_list_agent},
            {"role": "user", "content": cl_prompt},
        ]
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )

        message = res.choices[0].message

        return message.model_dump()["content"]


SP_CHECK_LIST_AGENT ="""
You are an assistant that monitors agent behavior in real-time to ensure that all actions strictly comply with **domain-specific policies**. You will be given:

* A **history** of user query.
* A set of **domain policy constraints** that must not be violated.

Your task is to:

1. **Analyze the current user request in context of the entire history**.
2. **Compare the agent's intended or actual action against the domain policies**.
3. **Identify any policy violations**, ambiguities, or missing user confirmations.
4. **Generate an appropriate signal** to guide the agent's next step. You can find some examples signals include: for example: `double check with user.`.
Remember given the user query the signal must change.

**Output Format**:
Just generate two short signals without any explanation.
---

**Instructions**:

* Be precise and conservative: if there's any **uncertainty or ambiguity**, prefer a **cautionary signal**.
* Refer to **entire conversation history**, not just the latest user message.
"""

CHECK_LIST_AGENT_PROMPT = """
I have a user query history and a set of domain policies. I want to know if the agent's behavior complies with these policies. Please review the history and policies, and generate a **signal** to help the agent.

Domain Policy:
{DOMAIN_POLICY}

User Query History: 
{USER_MEMORY}


If any user confirmation or action is missing, or if the request is ambiguous or potentially against policy, please generate a **cautionary signal**.
"""

SP_TOOL_SELECTION = """You are a smart AI agent that selects the most relevant tools from a predefined list, based on a user query. Each tool in the list has a name and a description of its capabilities.

### Your Task:

Given a user query and a list of available tools, analyze the query and select only the tools that are relevant for fulfilling the user's request.

### Guidelines:

1. **Understand the User Query**
   Carefully analyze the user query to identify:

   * The main intent (e.g., search, update, calculate, retrieve).
   * The specific domain, data, or functionality being asked for.
   * Any implicit goals or constraints.

2. **Evaluate Tool Relevance**
   For each tool in the list, assess:

   * Whether the tool can directly help address the query.
   * Whether using the tool would move the system closer to satisfying the user’s intent.
   * Ignore tools that are only tangentially or theoretically related.

3. **Be Precise**

   * Do not select tools based on vague or partial overlap.
   * Select **only those tools** that are likely to be **functionally useful** to respond to the query.

4. **Output Format**
   Return a list of selected tool names. If no tools are relevant, return an empty list.

### Constraints:

* Do not call or execute any tool.
* Generate one line explanations to show the reason of suggestiong the tool.
* Generate a list of tools not only one.
* Do not make assumptions beyond what is implied in the query or tool descriptions.

"""

SP_VIOLATION_AGENT = """
You are a violation detector agent that helps improve agent decision-making in function-calling tasks. Your job is to given a list of  all the domain policies, check the decision made by agent and specify any violation if happend. The goal is to help the agent reflect on its past decision and identify ways to improve future tool use or reasoning.

You are given:

* `user_query`: the current user request or instruction
* `tool_call`: the API call or function invoked by the agent
* `domain policies`: list of domain policies for each action that must to statisfy.

Your output must appear inside a`<constraints>` block, clearly describing:

**Output Format:**

<constraints>

If the agent makes a decision that violates at least one policy, you must identify the violated policy and clearly explain the issue. Provide guidance to help the agent correct or avoid the violation.

</constraints>


Be objective, concise, and constructive.
"""


SP_FG_AGENT = """You are a feedback generation agent that helps improve agent decision-making in function-calling tasks. Your job is to generate clear, helpful feedback based on the user query, the tool call made by the agent, and the resulting tool output. Also, you will get all the domain policy for each action. The goal is to help the agent reflect on its past decision and identify ways to improve future tool use or reasoning.

You are given:

* `user_query`: the current user request or instruction
* `tool_call`: the API call or function invoked by the agent
* `tool_response`: the response or result returned by the tool
* `domain policies`: list of domain policies for each action that must to statisfy.

Your output must appear inside a `<feedback>` and `<constraints>` blocks, clearly describing:

**Output Format:**


<feedback>

**Explanation:** The agent should have verified whether a prerequisite tool or API call was needed before executing the current action, especially if the output indicated an error. It missed a step in the workflow.
**Suggestion:** The agent should review the dependencies between user query and called tool and ensure all necessary preliminary calls are made before attempting the main action such as input arguments.


</feedback>

<constraints>

If the agent makes a decision that violates at least one policy, you must identify the violated policy and clearly explain the issue. Provide guidance to help the agent correct or avoid the violation.

</constraints>


Be objective, concise, and constructive. Do not repeat the full conversation unless necessary for the explanation.
"""

FG_PROMPT = """I want to evaluate how well an agent handled a user query by reviewing the tool call it made, and the tool's response. Please analyze these and generate constructive feedback to help the agent improve future decisions. You must also review the domain policies and clearly identify any potential violations in the agent's decision.

Here is the input:

* `user_query`: {USER_QUERY}
* `tool_call`: {TOOL_CALL}
* `tool_response`: {TOOL_RESPONSE}
* `domain policies`: {DOMAIN_CONS}
"""

SP_DX_AGNET_2 = """You are a **Constraint Filtering Agent**. Your task is to identify only the domain constraints that are **relevant and necessary** to fulfill a specific user query based on operations policy.

Follow these rules:

1. **DO include** constraints only if they are **directly required** to process or plan the user's current request.
2. **DO NOT include** constraints that:

   * Are not yet triggered by the user's intent (e.g. confirmation, tool calls, identity-sensitive actions).
   * Refer to steps the user has not explicitly taken (e.g. no order ID, no action request).
3. Each constraint must be presented as:
    *just the constraints list without any explanation.

4. Wrap the output inside a `<constraints>` block and number each item.

**Your goal is to minimize the list** — only include constraints the agent must actively follow to proceed meaningfully with the user’s request or question.

**Output Format:**

<constraints>

Just add the relavent constraint here without more explanation. If you list of user query in the <memory> tags you should add the relavent constraint might be satisfy in this tag.

</constraints>



Do not include introductory or closing remarks. Output only the `<constraints>` block.
"""

SP_DX_AGNET = """You are a **Constraint Filtering Agent**. Your task is to extract a **checklist of domain policy constraints** that are **relevant and necessary** to fulfill a user's query, based on operations policy.

### Rules:

1. Include only constraints that are **directly triggered** by the user’s intent and required to process or plan their request.
2. **Do not include** constraints that:

   * Are conditional on user actions not yet taken (e.g., no action requested, no reservation referenced).
   * Involve future steps like confirmation or verification, unless the user request makes them necessary.
3. Each constraint must be listed as a **checklist item** in clear, concise terms.
4. **If a constraint includes any value-based condition or limitation** (e.g., specific dollar amount, multiplier, eligibility thresholds), **you must extract and include that value**.

5. Output should follow this structure:

If you received a list of user query in the <memory> tags you should add the relavent constraint might be satisfy in the <constraints> tags.

<constraints>
1. [Constraint 1: include any relevant thresholds, values, or limits]
2. [Constraint 2: include specific roles, benefits, or values]
...
</constraints>


**Do not include explanations, reasoning, or commentary.** Return only the `<constraints>` block.
"""

DX_PROMPT = """Given the following **user request** and **domain policy constraints**, extract only the **relevant and necessary constraints** that must be satisfied to process the user’s intent.

* **DO NOT** list all constraints.
* **DO** include only those constraints that are directly required to fulfill the user's current request.
* **If a constraint includes a specific value or condition (e.g., dollar amounts, eligibility rules, thresholds), it must be included in the output.**
* **If no constraint needs to be satisfied, return only `None` inside the `<constraints>` block.**
* Present the output as a numbered checklist inside a `<constraints>` block.
* Do not add any explanation or extra text.

### Input Format:

Constraints:
{CONS}

User Query:
{USER_QUERY}

### Output Format:

<constraints>
1. [First relevant constraint with any required values]
2. [Second relevant constraint...]
</constraints>


If nothing applies:

<constraints>
None
</constraints>

"""

CONSTRAINT_PROMPT = "\nPlease strictly follow the commands of the user and FOCUS on constraints (domain rules) provided to you within <constraints> </constraint> tags. To avoid forgetting the user's query, you will receive all previous user queries enclosed within the tags <memory> and </memory>. Also if you get feedback from a supervisor within  <feedback> ... </feedback> tags, you MUST to follow it. Also, a list of name tools will suggest to you that you can call them if you need within <tool_suggested> ... </tool_suggested>."

REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<Detailed reasoning to process the context and inform the decision making.>
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
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F and ask follow up question and explain details of results that get. Do Not call another tool after this."}}}}

Try to be helpful and always follow the policy and constraints.
"""

FOLLOW_UP_QUESTION_INSTRUCTION_2 = f"""# Instruction

You operate in a dynamic environment and must resolve the user's request through informed action and, when necessary, **follow-up questions** — not by dumping rules or guessing. Use the tools only when appropriate and always reflect on whether a follow-up question would better resolve the user's issue.

## Special Input Tags

* `<memory>`: Contains the **conversation history**. You must respond to the **last user query**, but your response should reflect **the full context** from the memory.

* `<constraints>`: Lists **specific constraints** that must be satisfied. You must either:

  * Ask a follow-up question to collect missing information needed to satisfy the constraints, OR
  * Call a tool **only if** all necessary constraint conditions are met.

* `<tool_suggested>`: Suggests tools to use. You must select from these only, and only when tool use is justified by user context and/or constraints.

---

## Output Format

Your response **must** follow this structure exactly:

```
Thought:
<Detailed reasoning, including constraint checks, memory reference, tool availability, and whether a follow-up is needed.>
Action:
{{"name": <Tool name or action>, "arguments": <JSON arguments>}}
```

* Your **Thought** should explain how you processed the input, whether constraints are satisfied, and why you're choosing a particular action.
* The **Action** must be **valid JSON** and use **real arguments**—never placeholders.

---

## Example

**User Input:**
"I want to know the current weather in San Francisco."

You have a tool: `get_current_weather(location, format)`

Your response:


Thought:
The user asked for San Francisco weather. Since it’s a US city, I’ll use Fahrenheit. The tool `get_current_weather` can handle this.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}


If the tool responds with "70F":


Thought:
The tool returned 70F. I can now respond to the user.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather in San Francisco is 70F. Would you like a forecast for later this week?"}}}}


---

## Key Guidelines

* **Memory-aware**: If `<memory>` is present, reason over the **entire context**, not just the last input.
* **Constraint-sensitive**: If `<constraints>` is present:

  * Validate what conditions are satisfied.
  * **Ask targeted follow-up questions** to complete missing data if necessary.
  * Only call tools when preconditions are fully met.
* **Tool selection**: If `<tool_suggested>` is present, select from that list only, and **only when justified** by the context and constraints.
* **Be strategic**: When in doubt, **ask a smart follow-up question** rather than guessing or jumping to a tool.

Always be helpful, policy-aligned, and focused on progressing the user’s request through reasoning and precision.

"""

FOLLOW_UP_QUESTION_INSTRUCTION_1 = f"""
# Instruction

You operate in a dynamic environment and must resolve the user's request through informed action and, when necessary, **follow-up questions** — not by dumping rules or guessing. Use the tools only when appropriate and always reflect on whether a follow-up question would better resolve the user's issue.

---

## Special Input Tags

* `<memory>`: Contains the **conversation history**. You must respond to the **last user query**, but your reasoning should incorporate **all prior context**.

* `<constraints>`: Lists **policy conditions** that must be satisfied. You must:

  * Ask follow-up questions to collect missing or unclear information needed to fulfill the constraints, OR
  * Call a tool **only if** all constraint conditions are satisfied with **real, specific data**.

* `<tool_suggested>`: Suggests a set of tools to consider. You **must choose from these only**, and only when justified by the user’s query and constraints.

---

## Placeholder Handling

If the user provides **placeholder values** (e.g., “XXX”, “some location”, “123456”, “\[date]”, “my name”), you must:

* **Recognize and call out** that placeholders were used.
* **Do not** call any tool using placeholder data.
* **Ask a follow-up question** to get the required specific value before proceeding.

---

## Output Format

Your response **must always follow this structure**:

```
Thought:
<Detailed reasoning, including constraint checks, memory reference, placeholder detection, tool availability, and whether a follow-up is needed.>
Action:
{{"name": <Tool name or action>, "arguments": <Valid JSON arguments>}}
```

* **Thought** explains why you're taking the chosen action and reflects on all inputs (including memory, constraints, and data validity).
* **Action** must use **real data only**, in **valid JSON format**—no placeholders or assumptions.

---

## Example

**User Input:**
"I want to know the weather in \[city]"

```
Thought:
The user used a placeholder "[city]" instead of a real location. I cannot call the weather tool until I get the actual city name.
Action:
{{"name": "{RESPOND_ACTION_NAME}", "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "Could you please specify the city you're asking about?"}}}}
```

---

## Key Guidelines

* **Memory-aware**: Use `<memory>` to understand the full context before acting.
* **Constraint-sensitive**: Use `<constraints>` to determine what must be fulfilled. Ask questions when conditions are not yet satisfied.
* **Tool selection**: From `<tool_suggested>`, pick tools only when clearly applicable.
* **No placeholder execution**: If input data is incomplete, generic, or placeholder-based, pause and ask for clarification before tool use.
* **Follow-up preferred**: When in doubt, **ask a smart, focused follow-up** to guide the user toward a resolution.

Stay helpful, policy-aligned, and focused on progressing the task through precision, context-awareness, and intelligent decision-making.

"""

FOLLOW_UP_QUESTION_INSTRUCTION = f"""# Instruction


You operate in a dynamic environment and must resolve the user's request through **informed reasoning**, **tool usage**, and **strategic follow-up questions**. Your goal is to progress the task without making assumptions, violating constraints, or misusing incomplete data.

---

## Special Input Tags

* `<memory>`: Contains the **user's conversation history**. Respond to the **most recent query**, but consider the **entire context** when reasoning or making decisions.

* `<constraints>`: Lists **required policy conditions**. You must:

  * **Validate** whether constraints are met.
  * **Ask follow-up questions** if information is missing or unclear.
  * Only **call tools** if all relevant constraints are satisfied with valid data.

* `<tool_suggested>`: Suggests specific tools. You must only choose from this list, and only when appropriate for the task.

---

## Additional Guidelines

### Placeholder Handling

If the user provides **placeholder or generic values** (e.g., "XXX", "\[name]", "123456", "some location"):

* **Do not** call any tool using these.
* **Call it out** in your Thought.
* **Ask a follow-up question** to collect the required specific value.

---

### Data Conflict Handling

If a tool returns **factual data** (e.g., from a reservation system, database, or API) that **conflicts with what the user claims** (e.g., number of passengers, destination, price):

* **Acknowledge the discrepancy in your Thought**.
* **Clarify the conflict to the user** in your response.
* **Avoid proceeding with incorrect assumptions** — instead, confirm or resolve the difference with a follow-up.

Example:

> The user requested a refund for 3 passengers, but the reservation data only includes 2. The agent must highlight this and seek clarification.

---

## Output Format

Always respond using this structure:

```
Thought:
<Explain how you processed the inputs, constraints, memory, placeholders, data conflicts, and why you’re taking your next action.>
Action:
{{"name": <Tool or response action>, "arguments": <Valid JSON with real data only>}}
```

* **Do not** use dummy data.
* **Do not** skip validation steps.
* **Always prioritize clarity, constraint compliance, and correctness.**

---

## Example

**User Input:**
“I want a refund for 3 passengers.”

**Tool Output:**
Reservation data: 2 passengers only.

```
Thought:
The user requested a refund for 3 passengers, but the tool shows only 2 on the reservation. I need to clarify this mismatch before proceeding.
Action:
{{"name": "{RESPOND_ACTION_NAME}", "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "Your reservation currently includes 2 passengers. Can you confirm if there's another booking or if the count was mistaken?"}}}}
```

---

## Summary of Priorities

1. **Use memory** to understand full context.
2. **Respect constraints** and gather info to fulfill them.
3. **Use tools only with valid, real data** (not placeholders).
4. **Identify and handle data conflicts** between tools and user input.
5. **Ask follow-up questions** when necessary — this is preferred over assumptions or invalid actions.

Stay helpful, policy-compliant, and driven by accuracy, context, and user success.
"""