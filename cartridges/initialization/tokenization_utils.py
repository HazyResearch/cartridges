from typing import Optional
import torch


def llama3_tokenize_data_into_system_prompt(
    tokenizer,
    content: str,
    max_tokens: Optional[int],
) -> torch.Tensor:
    BOS_TOKEN_ID= 128000
    EOS_TOKEN_ID = 128009

    START_HEADER_ID = 128006
    END_HEADER_ID = 128007
    SYSTEM_ID = 9125

    input_ids = tokenizer.apply_chat_template([{"role": "system", "content": content}])
    assert input_ids[-1] == EOS_TOKEN_ID

    if max_tokens is not None and len(input_ids) > max_tokens:
        input_ids = input_ids[: max_tokens - 1] + [EOS_TOKEN_ID]

    assert input_ids[:4] == [BOS_TOKEN_ID, START_HEADER_ID, SYSTEM_ID, END_HEADER_ID]

    return torch.tensor(input_ids)[None, :]



def qwen_tokenize_data_into_system_prompt(
    tokenizer,
    content: str,
    max_tokens: Optional[int],
) -> torch.Tensor:
    END_TOKEN_IDS = [151645, 198]

    input_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": content}],
        include_special_tokens=True,
    )

    if max_tokens is not None and len(input_ids) > max_tokens:
        input_ids = input_ids[: max_tokens - len(END_TOKEN_IDS)] + END_TOKEN_IDS
    

    return torch.tensor(input_ids)[None, :]


MODEL_TO_SYSTEM_PROMPT_TOKENIZER = {
    "meta-llama/Llama-2-7b-chat-hf": llama3_tokenize_data_into_system_prompt,
    "meta-llama/Llama-3.2-1B-Instruct": llama3_tokenize_data_into_system_prompt,
    "meta-llama/Llama-3.2-3B-Instruct": llama3_tokenize_data_into_system_prompt,
    "meta-llama/Llama-3.1-8B-Instruct": llama3_tokenize_data_into_system_prompt,
    "Qwen/Qwen3-0.6b": qwen_tokenize_data_into_system_prompt,
    "Qwen/Qwen3-1.7b": qwen_tokenize_data_into_system_prompt,
    "Qwen/Qwen3-4b": qwen_tokenize_data_into_system_prompt,
    "Qwen/Qwen3-8b": qwen_tokenize_data_into_system_prompt,
    "Qwen/Qwen3-14b": qwen_tokenize_data_into_system_prompt,
    "Qwen/Qwen3-32b": qwen_tokenize_data_into_system_prompt,
}
MODEL_TO_SYSTEM_PROMPT_TOKENIZER = {k.lower(): v for k, v in MODEL_TO_SYSTEM_PROMPT_TOKENIZER.items()}


CARTRIDGES_LLAMA3_CHAT_TEMPLATE = """
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

MODEL_TO_CHAT_TEMPLATE = {
    "meta-llama/Llama-3.2-3B-Instruct": CARTRIDGES_LLAMA3_CHAT_TEMPLATE,
}


MODELS_WITH_THINKING = {
    "Qwen/Qwen3-0.6b",
    "Qwen/Qwen3-1.7b",
    "Qwen/Qwen3-4b",
    "Qwen/Qwen3-8b",
    "Qwen/Qwen3-14b",
    "Qwen/Qwen3-32b",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "zai-org/GLM-4.5",
    "zai-org/GLM-4.5-Air",
    "zai-org/GLM-4.5-FP8",
    "zai-org/GLM-4.6",
    "zai-org/GLM-4.6-FP8",
    "zai-org/GLM-4.7",
}