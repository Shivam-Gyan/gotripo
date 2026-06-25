from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from supervisor import create_supervisor_agent


load_dotenv()

st.set_page_config(
    page_title="GoTripo Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_state() -> None:
    defaults = {
        "messages": [],
        "turn_cache": {},
        "thread_id": str(uuid.uuid4()),
        "model_name": "groq:openai/gpt-oss-120b",
        "temperature": 0.7,
        "use_memory": True,
        "show_trace": True,
        "quick_prompt": "",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def get_supervisor(model_name: str, temperature: float, use_memory: bool):
    return create_supervisor_agent(
        model_name=model_name,
        temperature=temperature,
        use_memory=use_memory,
    )


def normalize_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2, ensure_ascii=True)
        return str(content)

    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return str(message)


def cache_key(prompt: str, history: list[dict[str, str]], model_name: str, temperature: float, use_memory: bool) -> str:
    payload = {
        "prompt": prompt,
        "history": history,
        "model_name": model_name,
        "temperature": temperature,
        "use_memory": use_memory,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_tool_trace(messages: list[Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            trace.append(
                {
                    "kind": "call",
                    "name": call.get("name", "tool"),
                    "arguments": call.get("args", {}),
                }
            )

        if message.__class__.__name__.endswith("ToolMessage"):
            trace.append(
                {
                    "kind": "result",
                    "name": getattr(message, "name", "tool"),
                    "content": normalize_text(message),
                }
            )

    return trace


def render_tool_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.info("No specialist tools were exposed for this turn.")
        return

    for index, item in enumerate(trace, 1):
        if item["kind"] == "call":
            st.markdown(
                f"""
                <div class="trace-card trace-call">
                    <div class="trace-title">Step {index}: {item['name']}</div>
                    <div class="trace-meta">Tool call arguments</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(json.dumps(item["arguments"], indent=2, ensure_ascii=True), language="json")
        else:
            st.markdown(
                f"""
                <div class="trace-card trace-result">
                    <div class="trace-title">Step {index}: {item['name']} result</div>
                    <div class="trace-meta">Returned to the supervisor</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(item["content"])


def sidebar_controls() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### Control Panel")
        st.caption("Tune the agent and session behavior.")

        st.session_state.model_name = st.text_input("Model", value=st.session_state.model_name)
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=float(st.session_state.temperature),
        )
        st.session_state.use_memory = st.toggle("Enable conversation memory", value=bool(st.session_state.use_memory))
        st.session_state.show_trace = st.toggle("Show tool trace", value=bool(st.session_state.show_trace))

        st.divider()
        destination = st.text_input("Destination hint", value="Tokyo")
        traveler_type = st.selectbox("Traveler type", ["solo", "couples", "families", "business", "luxury", "budget"], index=1)
        budget = st.number_input("Budget", min_value=0.0, value=800.0, step=50.0)

        st.divider()
        st.markdown("### Quick prompts")
        quick_prompts = [
            f"Find hotels in {destination} for a {traveler_type} traveler under ${int(budget)}.",
            f"Plan a complete {destination} trip for a {traveler_type} traveler under ${int(budget)}.",
            f"Show flights, hotels, and activities for {destination}.",
        ]
        chosen_prompt = st.radio("", quick_prompts, label_visibility="collapsed")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("New chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.turn_cache = {}
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
        with col_b:
            if st.button("Refresh session", use_container_width=True):
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

    return {
        "destination": destination,
        "traveler_type": traveler_type,
        "budget": budget,
        "chosen_prompt": chosen_prompt,
    }


def run_turn(prompt: str) -> dict[str, Any]:
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
    ]
    key = cache_key(
        prompt=prompt,
        history=history,
        model_name=st.session_state.model_name,
        temperature=float(st.session_state.temperature),
        use_memory=bool(st.session_state.use_memory),
    )

    cached = st.session_state.turn_cache.get(key)
    if cached is not None:
        result = dict(cached)
        result["cached"] = True
        return result

    supervisor = get_supervisor(
        st.session_state.model_name,
        float(st.session_state.temperature),
        bool(st.session_state.use_memory),
    )

    payload = {"messages": history + [{"role": "user", "content": prompt}]}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    response = supervisor.invoke(payload, config=config)

    messages = response.get("messages", []) if isinstance(response, dict) else []
    final_message = messages[-1] if messages else response
    assistant_text = normalize_text(final_message)
    trace = extract_tool_trace(messages)

    result = {
        "assistant_text": assistant_text,
        "trace": trace,
        "cached": False,
    }
    st.session_state.turn_cache[key] = result
    return result


ensure_state()
sidebar = sidebar_controls()

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 174, 66, 0.18), transparent 26%),
                radial-gradient(circle at top right, rgba(58, 134, 255, 0.14), transparent 22%),
                linear-gradient(180deg, #0b1020 0%, #111827 100%);
            color: #eef2ff;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.25rem;
        }

        .hero-card {
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(17, 24, 39, 0.74);
            backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 16px 45px rgba(0, 0, 0, 0.25);
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(255, 255, 255, 0.12);
            color: white;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        .trace-card {
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: white;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }

        .trace-call {
            border-left: 5px solid #0ea5e9;
        }

        .trace-result {
            border-left: 5px solid #10b981;
        }

        .trace-title {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }

        .trace-meta {
            color: #475569;
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">Agentic travel workspace</div>
        <h1 style="margin:0; font-size:2.25rem; line-height:1.05;">GoTripo Studio</h1>
        <p style="margin:0.55rem 0 0; max-width:70ch; color:#cbd5e1;">
            Chat with a travel supervisor that routes requests to flights, hotels, activities, and itinerary specialists.
            The interface caches turns, preserves a conversation thread, and shows the tool path behind each answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    if not st.session_state.messages:
        st.markdown("### Try a prompt")
        st.info(sidebar["chosen_prompt"])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and st.session_state.show_trace:
                with st.expander("Tool calling trail", expanded=False):
                    render_tool_trace(message.get("trace", []))
                if message.get("cached"):
                    st.caption("Served from cache")

    prompt = st.chat_input("Ask for flights, hotels, activities, or a full travel plan")
    prompt = prompt or (sidebar["chosen_prompt"] if not st.session_state.messages else None)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking through the trip..."):
                turn = run_turn(prompt)

            st.markdown(turn["assistant_text"])
            if st.session_state.show_trace:
                with st.expander("Tool calling trail", expanded=False):
                    render_tool_trace(turn["trace"])
            if turn.get("cached"):
                st.caption("Served from cache")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": turn["assistant_text"],
                "trace": turn["trace"],
                "cached": turn["cached"],
            }
        )

with right_col:
    st.subheader("Session")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Cache entries", len(st.session_state.turn_cache))
    st.metric("Thread ID", st.session_state.thread_id[:8])
    st.metric("Memory", "On" if st.session_state.use_memory else "Off")

    st.divider()
    st.markdown("### What this UI shows")
    st.markdown(
        """
        - Cached supervisor responses for repeated turns
        - A chat-style conversation flow
        - A visible tool-calling trace for each assistant turn
        - Sidebar controls for model, memory, and prompt presets
        """
    )
