from llm_client import get_llm

print("Test connessione LLM locale...")

llm = get_llm(temperature=0.7)
response = llm.invoke("Dimmi 'Hello' se mi senti")

print(f"Risposta: {response.content}")
print(f"Token usati: {response.response_metadata}")