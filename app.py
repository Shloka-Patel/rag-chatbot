"""College Document Q&A chatbot website. Run: streamlit run app.py"""
import os
import tempfile

import streamlit as st

try:  # on Streamlit Cloud (Step 9) the key will live in st.secrets
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass  # in Codespaces the key already comes from your Codespaces secret

import rag  # noqa: E402  (imported after the key is set)

# ======================= YOUR CHOICES =======================
# Replace every "..." below with your own words.
APP_TITLE = "AskIt"
APP_DESCRIPTION = "Curious about Symbiosis Institute of Technology? Well, you've come to the right place. Ask our chatbot any questions about admissions, placements, programmes offered, etc. here."
DISCLAIMER = "If you find any difficulty getting your answers, feel free to contact us anytime."
EXAMPLE_QUESTIONS = [
    "What is the minimum attendance required?",
    "Is there any extra fee for a Minor or Honours specialisation?",
    "What happens if I enter the classroom late?", 
    "What internship options are part of the placement process?", 
]
SHOW_PASSAGE_TEXT = True   # True = show the matching text under each source, False = file and page only
# ============================================================

st.set_page_config(page_title=APP_TITLE, page_icon="🎓")

# Memory that survives Streamlit's re-runs
if "messages" not in st.session_state:
    st.session_state.messages = []      # the chat history


def loaded_documents():
    """Names of the PDFs currently stored in the database."""
    metadatas = rag.get_collection().get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metadatas})


# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("About")
    st.write(APP_DESCRIPTION)
    st.info(DISCLAIMER)

    st.subheader("Documents loaded")
    docs = loaded_documents()
    if docs:
        for name in docs:
            st.caption(f"📄 {name}")
    else:
        st.caption("No documents yet. Upload some below.")

    st.subheader("Try asking")
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example_{i}"):
            st.session_state.pending_question = example

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Add documents"):
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if files and st.button("Index documents"):
            for f in files:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(f.getvalue())
                with st.spinner(f"Indexing {f.name}..."):
                    n = rag.index_pdf(tmp.name, source=f.name)
                st.success(f"{f.name}: {n} chunks")
            st.rerun()


# ----------------------- Main area -----------------------
st.title(APP_TITLE)


def show_sources(sources):
    """Show which file and page each answer came from."""
    if not sources:
        return
    with st.expander(f"Sources used ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}**, page {s['page']}  ·  distance {s['distance']}")
            if SHOW_PASSAGE_TEXT:
                st.caption(s["text"][:300] + "...")


# Re-draw the conversation so far (Streamlit re-runs the whole script on every click)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

# A new question comes either from the chat box or from an example button
question = st.chat_input("Ask about attendance, courses, fees, placements...")
if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            reply, sources = rag.answer(question)
        st.markdown(reply)
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})