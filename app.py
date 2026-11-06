"""Streamlit chat app: Scholarship and Financial Aid Assistant."""
import hmac
import time

import streamlit as st
from src.links import official_link
from src import config
from src.documents import add_document, delete_document
from src.embeddings import EmbeddingError
from src.loaders import DocumentLoadError
from src.rag import AnswerError, answer_question
from src.security import UploadError
from src.vectorstore import get_collection, list_documents

st.set_page_config(page_title="Scholarship Assistant", page_icon="🎓")

EXAMPLES = [
    "What documents are required for post matric scholarship?",
    "What is the income limit for minority fee reimbursement?",
    "Which scholarships does AICTE offer for girl students?",
]
CATEGORIES = ["state", "central", "corporate", "institution", "other"]


def admin_login_ok(entered: str) -> bool:
    return bool(config.ADMIN_PASSWORD) and hmac.compare_digest(
        entered.encode("utf-8"), config.ADMIN_PASSWORD.encode("utf-8")
    )


def render_message(msg: dict, show_debug: bool):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return

        items, seen = [], set()
        for s in msg.get("sources") or []:
            label, url = official_link(s.get("source", ""))
            if url:
                if url in seen:
                    continue
                seen.add(url)
                items.append(f"[{label}]({url})")
            else:
                name = s.get("name") or s.get("source") or "Document"
                if config.SHOW_PAGE_NUMBERS and s.get("pages"):
                    pages = ", ".join(str(p) for p in s["pages"])
                    name += f", page{'s' if len(s['pages']) > 1 else ''} {pages}"
                items.append(name)
        if items:
            with st.expander(f"Verify on the official website ({len(items)})"):
                for item in items:
                    st.markdown(f"- {item}")

                if config.SHOW_SCREENSHOT_NOTE and msg.get("uses_screenshot"):
                    st.caption("⚠️ Based partly on website screenshots. Please confirm the details on the official portal.")
        if show_debug and msg.get("debug"):
            st.caption(msg["debug"])

def admin_panel():
    with st.expander("Administrator"):
        if not config.ADMIN_PASSWORD:
            st.caption("Admin tools are off. Set ADMIN_PASSWORD in the .env file to turn them on.")
            return
        if not st.session_state.get("admin_ok"):
            entered = st.text_input("Admin password", type="password")
            if st.button("Log in"):
                if admin_login_ok(entered):
                    st.session_state.admin_ok = True
                    st.rerun()
                else:
                    time.sleep(1)
                    st.error("Wrong password.")
            return

        if st.button("Log out"):
            st.session_state.admin_ok = False
            st.rerun()

        st.subheader("Add or replace a document")
        with st.form("upload_form", clear_on_submit=True):
            file = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
            scheme = st.text_input("Scheme name (optional)")
            provider = st.text_input("Provider (optional)")
            year = st.text_input("Academic year, e.g. 2026-27 (optional)")
            category = st.selectbox("Category", CATEGORIES)
            submitted = st.form_submit_button("Upload and index")
        if submitted:
            if file is None:
                st.error("Choose a file first.")
            else:
                meta = {"scheme_name": scheme, "provider": provider,
                        "academic_year": year, "category": category}
                try:
                    with st.spinner("Reading and indexing the document..."):
                        name, status = add_document(file.name, file.getvalue(), meta)
                    st.success(f"{name}: {status}.")
                except (UploadError, DocumentLoadError, EmbeddingError) as exc:
                    st.error(str(exc))
                except OSError as exc:
                    st.error(f"A file could not be saved. Close it if it is open in another program. ({exc})")

        docs = list_documents()
        st.subheader("Remove a document")
        if docs:
            choice = st.selectbox("Document", [d["source"] for d in docs])
            confirm = st.checkbox("Yes, remove it from the chatbot")
            if st.button("Remove") and confirm:
                try:
                    count = delete_document(choice)
                    st.success(f"Removed {choice} ({count} chunks). The file was moved to data/removed.")
                    st.rerun()
                except OSError as exc:
                    st.error(f"Could not move the file ({exc}).")

        docs = list_documents()
        st.subheader("Indexed documents")
        st.caption(f"{len(docs)} documents, {sum(d['chunks'] for d in docs)} chunks")
        for d in docs:
            st.markdown(f"- `{d['source']}`, {d['chunks']} chunks, {d['reliability']}")


def sidebar() -> bool:
    with st.sidebar:
        st.header("About")
        st.write(
            "Ask about scholarships, eligibility, required documents, deadlines and how "
            "to apply. Answers come only from the scholarship documents loaded here, "
            "with the file and page shown."
        )
        show_debug = st.checkbox("Show how my question was searched (testing)")
        if st.button("Clear chat"):
            st.session_state.messages = []
        with st.expander("Privacy and limits"):
            st.markdown(
                "- Do not type personal details such as Aadhaar, phone or bank numbers.\n"
                "- Your question is sent to Google's Gemini API to find and write the "
                "answer. On Google's free tier, submitted content may be used to improve "
                "Google's products.\n"
                "- This app does not save your chat. It disappears when you close or "
                "refresh the page.\n"
                "- The documents are public scholarship documents stored on the server.\n"
                "- Information changes every year. Confirm deadlines and amounts on the "
                "official portal. This assistant cannot guarantee that you will get a "
                "scholarship."
            )
        admin_panel()
    return show_debug


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    show_debug = sidebar()

    st.title("🎓 Scholarship Assistant")
    st.caption("Guidance from official scholarship documents. Not a guarantee of approval.")

    if not config.GEMINI_API_KEY:
        st.error("The Gemini API key is missing. Add GEMINI_API_KEY to the .env file and restart.")
        st.stop()
    try:
        chunk_count = get_collection().count()
    except Exception:
        st.error("The document database could not be opened. Check the data/chroma folder.")
        st.stop()
    if chunk_count == 0:
        st.warning("No documents are loaded yet. An administrator needs to add some.")

    for msg in st.session_state.messages:
        render_message(msg, show_debug)

    if not st.session_state.messages:
        st.write("Try one of these:")
        for example in EXAMPLES:
            if st.button(example):
                st.session_state.pending_question = example
                st.rerun()

    pending = st.session_state.pop("pending_question", None)
    typed = st.chat_input("Ask about scholarships, eligibility, documents or deadlines...")
    question = pending or typed
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg, show_debug)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                result = answer_question(question, history)
            except AnswerError as exc:
                st.error(str(exc))
                return
        reply = {
            "role": "assistant",
            "content": result.text,
            "sources": result.sources,
            "citations_verified": result.citations_verified,
            "uses_screenshot": result.uses_screenshot,
            "debug": f"Searched as: {result.standalone_question} | top similarity {result.top_similarity:.2f}",
        }
    st.session_state.messages.append(reply)
    st.rerun()


main()