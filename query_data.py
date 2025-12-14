import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--show-sources", action="store_true", help="Show source documents.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text, show_sources=args.show_sources)


def query_rag(query_text: str, show_sources: bool = False):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = OllamaLLM(model="mistral")
    # response_text = model.invoke(prompt)
    
    # For testing purposes without Ollama, return the context as response
    response_text = f"Based on the provided context, here's what I found:\n\n{context_text[:1000]}..."

    print(f"Response: {response_text}")
    
    if show_sources:
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        print(f"Sources: {sources}")
    
    return response_text


if __name__ == "__main__":
    main()
