from typing import Callable, Tuple, Annotated
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.language_models import BaseLanguageModel

PROMPT_TEMPLATE = """Below is the context retrieved from the required file based on your query.
If you can't answer the question with or without the current context, you should try using a more refined search query according to your requirements, or ask for more contexts.

Your current query is: {question}

Retrieved context is:
{context}

Answer:"""

def get_rag_function(
    retriever: VectorStoreRetriever,
    llm: BaseLanguageModel,
    description: str = "",
) -> Tuple[Callable[[str, int], str], RetrievalQA]:
    """Creates a LangChain-style RAG function with a retriever and LLM."""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    def retrieve_content(
        message: Annotated[str, "Refined query for retrieving and answering."],
        n_results: Annotated[int, "Number of results to retrieve"] = 3,
    ) -> str:
        retriever.search_kwargs["k"] = n_results
        return qa_chain.run(message)

    retrieve_content.__doc__ = (
        description or "Retrieve content from documents for QA/codegen using LangChain."
    )

    return retrieve_content, qa_chain
