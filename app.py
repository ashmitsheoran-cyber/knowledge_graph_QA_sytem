import chainlit as cl
from agent_engine import agent_app

@cl.on_chat_start
async def on_chat_start():
    # This runs when the user opens the page
    await cl.Message(
        content="Welcome to the Hybrid RAG System. I can search our internal documents or browse the web for answers. How can I help you today?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content

    # 1. Create a "Thought Trace" Step in the UI
    async with cl.Step(name="Agent Routing & Retrieval", type="run") as step:
        step.input = query
        
        initial_state = {
            "query": query,
            "context": [],
            "answer": "",
            "source_type": "",
            "iterations": 0
        }

        # 2. Run the LangGraph Agent asynchronously so it doesn't freeze the UI
        # We use cl.make_async to wrap your synchronous LangGraph invoke
        result = await cl.make_async(agent_app.invoke)(initial_state)

        # 3. Update the Thought Trace based on what the agent did
        source = result.get('source_type', 'Unknown')
        iters = result.get('iterations', 0)

        actual_iters = min(iters, 3)  # judge sets iterations=4 on completion; cap display at 3
        if source == "Local":
            step.output = "Answer retrieved from local knowledge (internal vault or web cache)."
        else:
            step.output = f"Web research completed across {actual_iters} iteration(s)."

    # 4. Format the final output with a "Badge" for the source
    final_answer = result['answer']

    if source == "Local":
        badge = "**[LOCAL KNOWLEDGE]**\n\n"
    else:
        badge = "**[WEB RESEARCH]**\n\n"

    # 5. Send the final answer back to the user
    await cl.Message(content=badge + final_answer).send()