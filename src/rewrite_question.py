from langchain_core.messages import HumanMessage, convert_to_messages
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from src.config import get_response_model
from src.message_utils import get_current_question


class RewrittenQuestion(BaseModel):
    rewritten_question: str = Field(
        description="A concise rewritten search question without explanation."
    )


REWRITE_PROMPT = (
    "Rewrite the user's question to improve retrieval quality.\n"
    "Return only the rewritten question.\n"
    "Do not include analysis, reasoning, labels, markdown, or extra text.\n"
    "Original question: {question}"
)


def rewrite_question(state: MessagesState):
    """Rewrite the current user question into a cleaner retrieval query."""
    question = get_current_question(state["messages"])
    prompt = REWRITE_PROMPT.format(question=question)
    structured_model = get_response_model().with_structured_output(
        RewrittenQuestion,
        method="function_calling",
        strict=True,
    )
    response = structured_model.invoke([{"role": "user", "content": prompt}])
    rewritten_question = response.rewritten_question.strip() or question
    return {"messages": [HumanMessage(content=rewritten_question)]}


if __name__ == "__main__":
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    }

    response = rewrite_question(input)
    print(response["messages"][-1].content)
