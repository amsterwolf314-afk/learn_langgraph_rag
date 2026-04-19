from langchain_core.messages import BaseMessage


def get_original_question(messages: list[BaseMessage]) -> str:
    """Return the first human-authored question for display/debugging."""
    for message in messages:
        if message.type == "human":
            return message.content
    return messages[0].content


def get_current_question(messages: list[BaseMessage]) -> str:
    """Return the latest human-authored question used for retrieval/answering."""
    for message in reversed(messages):
        if message.type == "human":
            return message.content
    return messages[0].content
