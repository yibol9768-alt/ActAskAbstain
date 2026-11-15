"""Route AgentAbstain's OpenAI-SDK harness and judge to the H20 DeepSeek vLLM (chat completions)."""
import os
from openai import AsyncOpenAI, OpenAI

BASE = os.environ.get("DS_BASE", "http://127.0.0.1:18003/v1")
KEY = os.environ["DEEPSEEK_KEY"]
os.environ["OPENAI_API_KEY"] = KEY
os.environ["OPENAI_BASE_URL"] = BASE


def patch_agents():
    from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
    set_default_openai_client(AsyncOpenAI(base_url=BASE, api_key=KEY), use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)


def patch_judge():
    from abstention_factory.src.utils import call_llm

    def _call_openai(messages, model, temperature, max_tokens):
        client = OpenAI(base_url=BASE, api_key=KEY)
        r = client.chat.completions.create(model=model, messages=messages,
                                           temperature=temperature or 0.0,
                                           max_tokens=max(max_tokens, 4096))
        u = r.usage
        if u:
            call_llm._record_usage(model, u.prompt_tokens, u.completion_tokens)
        return r.choices[0].message.content or ""

    call_llm._call_openai = _call_openai
