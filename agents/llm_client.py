from langchain_openai import ChatOpenAI

def get_llm(temperature=0.7):
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        model="local-model",
        temperature=temperature,
        max_tokens=2048
    )