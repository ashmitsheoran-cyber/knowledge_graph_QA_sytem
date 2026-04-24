import sys
import io
import time
import asyncio
import chainlit as cl
from agent_engine import agent_app

# ── Capture agent's print output to build the thought trace ──────────────────
class StdoutCapture:
    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = self._buf = io.StringIO()
        return self._buf
    def __exit__(self, *args):
        sys.stdout = self._old

def parse_trace(log: str) -> list:
    steps = []
    for line in log.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('[ANALYST]'):
            steps.append({'node': 'Analyst',    'msg': line[len('[ANALYST]'):].strip()})
        elif line.startswith('[RESEARCHER]'):
            steps.append({'node': 'Researcher', 'msg': line[len('[RESEARCHER]'):].strip()})
        elif line.startswith('[WRITER]'):
            steps.append({'node': 'Writer',     'msg': line[len('[WRITER]'):].strip()})
        elif line.startswith('[JUDGE]'):
            steps.append({'node': 'Judge',      'msg': line[len('[JUDGE]'):].strip()})
    return steps

# ── Welcome screen ────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=(
        "## Strategic Intelligence Assistant\n\n"
        "I combine your **private document vault** with **live web search** "
        "to deliver cited, verified answers — no hallucinations.\n\n"
        "Ask me anything. I'll tell you exactly where every answer came from."
    )).send()

# ── Main message handler ──────────────────────────────────────────────────────
@cl.on_message
async def main(message: cl.Message):
    query = message.content
    start_total = time.time()

    initial_state = {
        "query":       query,
        "context":     [],
        "answer":      "",
        "source_type": "",
        "iterations":  0,
        "gap":         ""
    }

    # Run the full agent pipeline, capturing all print output for the trace
    with StdoutCapture() as buf:
        result = await cl.make_async(agent_app.invoke)(initial_state)

    total_time = time.time() - start_total
    raw_log    = buf.getvalue()
    trace      = parse_trace(raw_log)

    source   = result.get('source_type', 'Web')
    iters    = min(result.get('iterations', 0), 3)
    contexts = result.get('context', [])

    has_local = any("Internal:" in c or "| SOURCE:" in c for c in contexts)
    is_hybrid = (source == "Web") and has_local

    # ── Thought Trace — one Chainlit Step per pipeline node ──────────────────
    seen_nodes    = {}
    ordered_nodes = []
    for step in trace:
        n = step['node']
        if n not in seen_nodes:
            seen_nodes[n] = []
            ordered_nodes.append(n)
        seen_nodes[n].append(step['msg'])

    for node in ordered_nodes:
        msgs = seen_nodes[node]
        async with cl.Step(name=node, type="run") as s:
            s.input  = query if node == 'Analyst' else msgs[0]
            s.output = '\n'.join(msgs)

    # ── Build header badge + timing ───────────────────────────────────────────
    if is_hybrid:
        badge     = "**[ HYBRID — VAULT + WEB ]**"
        time_note = f"*Vault + web combined in {total_time:.1f}s*"
    elif source == "Local":
        badge     = "**[ LOCAL VAULT ]**"
        time_note = f"*Local memory in {total_time:.1f}s*"
    else:
        attempt_str = f"{iters} search attempt{'s' if iters != 1 else ''}"
        badge       = "**[ WEB RESEARCH ]**"
        time_note   = f"*{attempt_str} in {total_time:.1f}s*"

    final_answer = result.get('answer', 'No answer was generated.')

    # ── Stream the answer token by token ─────────────────────────────────────
    msg = cl.Message(content="")
    await msg.send()

    # Stream the header first
    header = f"{badge}  {time_note}\n\n---\n\n"
    await msg.stream_token(header)

    # Stream the answer word by word
    words = final_answer.split(" ")
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        await msg.stream_token(token)
        await asyncio.sleep(0.012)  # ~80 words/sec — feels natural, not too fast or slow

    await msg.update()
