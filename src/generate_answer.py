from langgraph.graph import MessagesState

from src.config import get_response_model
from src.message_utils import get_current_question


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = get_current_question(state["messages"])
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = get_response_model().invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}
