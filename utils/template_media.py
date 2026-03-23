import json
import logging
import os
import re
import tempfile
import unicodedata

import numpy as np
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_DIR = os.getenv("RAG_DIR", os.path.join("outputs", "RAG"))

# Built once at import time, not on every call
_SUPPORTED_TYPES = {
    "text/plain": "txt",
    "text/markdown": "md",
    "text/html": "html",
    "text/csv": "csv",
    "text/rtf": "rtf",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "pptx",
    "application/json": "json",
    "application/rtf": "rtf",
}

_EXT_MAP = {
    ".txt": "txt", ".md": "md", ".pdf": "pdf",
    ".docx": "docx", ".doc": "docx",
    ".xlsx": "xlsx", ".xls": "xlsx",
    ".csv": "csv", ".pptx": "pptx", ".ppt": "pptx",
    ".json": "json", ".html": "html", ".htm": "html", ".rtf": "rtf",
}


# ── Prompt builders ────────────────────────────────────────────────────────────

def title_prompt(content: str) -> str:
    return (
        "Generate a short, concise title (3–6 words) that summarizes the following message.\n"
        "Output only the title as plain text. Do not include quotes, punctuation, commentary, or any additional words!\n"
        f"Only produce the title itself:\n{clean_text(str(content))}"
    )


def summary_prompt(content: str) -> str:
    return (
        "Generate a short, concise summary of the content found in the first 3000 characters of the provided text.\n"
        'If the text already contains a summary-like section (e.g., "abstract", "summary"), extract only that section instead.\n'
        "The summary must focus strictly on the subject matter, ideas, findings, or events described in the text.\n"
        "Do not focus on the author, their background, or personal details.\n"
        "Output only the summary/extracted text as plain text.\n"
        f"Only produce the summary/extracted text itself:\n{clean_text(str(content))}"
    )


def build_prompt_template(messages: list, mode: str, system_prompt: str = None, summerize: bool = False) -> list:
    def wrap(text):
        return [{"type": "text", "text": text}] if mode == "vlm" else text

    if summerize:
        return [{"role": msg.get("role"), "content": wrap(msg.get("text"))} for msg in messages]

    if not system_prompt or not system_prompt.strip():
        system_prompt = "You are a helpful assistant."

    prompt = [{"role": "system", "content": wrap(system_prompt)}]
    for msg in messages[:-1]:
        prompt.append({"role": msg.get("role"), "content": wrap(msg.get("text"))})

    last = messages[-1]
    prompt.append({"role": "user", "content": last.get("vlm_content") if mode == "vlm" else last.get("content")})
    return prompt


def summarize_history(messages: list, mode: str, model_name: str, max_window_size: int = 4096) -> list:
    if not messages:
        return messages

    from utils.api_client import APIClient

    template = build_prompt_template(messages=messages, mode=mode, summerize=True)
    if APIClient.count_tokens(model_name=model_name, text=template) < max_window_size:
        return messages

    placeholder = st.empty()
    placeholder.markdown("*Trying to keep up with the whole conversation...*")
    st.toast("The conversation became too long, the model will begin to have a hard time remembering all details.", duration="long")
    logger.info("Summarizing old conversation")

    query = (
        "Summarize the entire previous conversation for memory purposes.\n\n"
        "Instructions:\n"
        "- This summary is NOT a dialogue and must NOT be treated as chat turns.\n"
        "- Do NOT write responses or continue the conversation.\n"
        "- Only describe what the user said and what the assistant said.\n\n"
        "Output format (exactly):\n\n"
        "CONVERSATION SUMMARY\nUser:\n<concise summary of everything the user asked or stated>\n\n"
        "Assistant:\n<concise summary of everything the assistant explained or replied>\n\nEnd of summary."
    )
    template.append({"role": "user", "content": [{"type": "text", "text": query}] if mode == "vlm" else query})

    user_recall = "The following is a condensed memory of the prior conversation. Use it only as background context. Respond normally to the user's next message."
    response = APIClient.generate(model_name=model_name, template=template, max_tokens=1024)

    placeholder.empty()
    return [
        {"role": "user", "text": user_recall, "content": user_recall, "vlm_content": [{"type": "text", "text": user_recall}], "medias": [], "docs": []},
        {"role": "assistant", "vlm_content": [{"type": "text", "text": response}], "content": response, "text": response},
    ]


# ── RAG utilities ──────────────────────────────────────────────────────────────

def extract_keywords(texts: list, top_n: int = 5) -> list:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=120, min_df=1, max_df=0.6,
        token_pattern=r"\b(?:[A-Z]{2,}|\w{3,})\b",
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True,
        ngram_range=(1, 2), analyzer="word", lowercase=False,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        keywords_per_chunk = []
        for row in tfidf_matrix:
            scores = row.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords_per_chunk.append([feature_names[i] for i in top_indices if scores[i] > 0])
        keywords_per_chunk[-1].append("link")
        return keywords_per_chunk
    except Exception as e:
        logger.warning(f"TF-IDF failed: {e}")
        return [[w for w in text.split() if len(w) > 3][:top_n] for text in texts]


def cosine_similarity(a, b) -> float:
    a, b = np.array(a).flatten(), np.array(b).flatten()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_matching(collection, embedding_model, all_doc_ids: list, chunks: list, threshold: float = 0.90) -> bool:
    if not all_doc_ids or not chunks:
        return False

    new_avg = np.mean(embedding_model.encode(chunks), axis=0)

    for doc_id in all_doc_ids:
        try:
            data = collection.get(where={"doc_id": doc_id}, include=["embeddings"])
            if not data["embeddings"]:
                continue
            existing = np.array(data["embeddings"])
            existing_avg = existing if existing.ndim == 1 else np.mean(existing, axis=0)
            if cosine_similarity(new_avg, existing_avg) >= threshold:
                logger.info(f"Duplicate detected with {doc_id}")
                return True
        except Exception as e:
            logger.error(f"Error checking similarity for {doc_id}: {e}")

    return False


def hierarchical_retrieval(
    collection, embedding_model, chat_docs: list, query: str,
    top_n: int = 5, threshold: float = 0.7, user: str = "user",
) -> dict:
    file_path = os.path.join(RAG_DIR, "DOCUMENT_METADATA.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            document_metadata = json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        raise ValueError(f"Failed to load JSON: {file_path}")

    if not chat_docs:
        return {}

    chat_doc_metadata = {doc_id: meta for doc_id, meta in document_metadata.items() if doc_id in chat_docs}
    query_words = set(query.lower().split())
    query_embedding = embedding_model.encode([query])[0]

    doc_scores = []
    for doc_id, meta in chat_doc_metadata.items():
        summary = meta.get("summary", "")
        word_score = len(query_words & set(summary.lower().split())) / max(len(query_words), 1)
        semantic_score = cosine_similarity(query_embedding, embedding_model.encode([summary])[0])
        doc_scores.append((doc_id, 0.3 * word_score + 0.7 * semantic_score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    relevant_doc_ids = {
        doc_id: document_metadata[doc_id]["total_chunks"]
        for doc_id, score in doc_scores[:4]
        if score >= 0.3
    }

    if not relevant_doc_ids:
        return {}

    chunk_ids = [
        f"{doc_id}_chunk_{i}"
        for doc_id, total_chunks in relevant_doc_ids.items()
        for i in range(total_chunks)
    ]
    all_data = collection.get(where={"chunk_id": {"$in": chunk_ids}}, include=["metadatas", "documents"])

    if not all_data or not all_data.get("ids"):
        return {}

    fetched_chunk_ids = [meta["chunk_id"] for meta in all_data["metadatas"]]
    results = collection.query(
        query_texts=[query],
        where={"chunk_id": {"$in": fetched_chunk_ids}},
        n_results=min(50, len(fetched_chunk_ids)),
    )

    candidates = sorted(
        [
            {"id": results["ids"][0][i], "doc": results["documents"][0][i],
             "meta": results["metadatas"][0][i], "distance": d}
            for i, d in enumerate(results["distances"][0])
            if d <= threshold
        ],
        key=lambda x: x["distance"],
    )[:top_n]

    return {
        "ids": [[c["id"] for c in candidates]],
        "documents": [[c["doc"] for c in candidates]],
        "metadatas": [[c["meta"] for c in candidates]],
        "distances": [[c["distance"] for c in candidates]],
    }


# ── Media processing ───────────────────────────────────────────────────────────

def media_resize(file, max_width: int = 256, fps: int = 15):
    file_extension = os.path.splitext(file.name)[1]

    if file.type.startswith("image"):
        from PIL import Image

        img = Image.open(file).convert("RGB")
        if max(img.size) > max_width:
            img.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
            img.save(f, format="JPEG", quality=85, optimize=True)
            return f.name, "image"

    elif file.type.startswith("video"):
        import cv2

        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
            f.write(file.read())
            input_path = f.name

        cap = cv2.VideoCapture(input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / fps) if original_fps > fps else 1
        new_w = max_width if w > max_width else w
        new_h = int(h * (new_w / w))

        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                out.write(cv2.resize(frame, (new_w, new_h)))
            frame_count += 1

        cap.release()
        out.release()
        try:
            os.remove(input_path)
        except FileNotFoundError:
            pass

        return output_path, "video"


def clear_temp(list_of_path: list):
    for path in list_of_path:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


# ── Query builders ─────────────────────────────────────────────────────────────

def make_rag_query(text_content: str, relevant_chunks: dict) -> str:
    if not relevant_chunks or not relevant_chunks["ids"][0]:
        return text_content

    context = ""
    sources = []
    for i in range(len(relevant_chunks["ids"][0])):
        title = relevant_chunks["ids"][0][i].rsplit("_", 4)[0]
        if title not in sources:
            sources.append(title)
        context += f"{relevant_chunks['documents'][0][i]}\n\n"

    source_list = ", ".join(sources[:3])
    if len(sources) > 3:
        source_list += f" and {len(sources) - 3} more"

    return clean_text(
        f"<context>\nThe user has access to uploaded documents: {source_list}\n"
        f"Relevant excerpts:\n{context.strip()}\n</context>\n"
        f"<user_message>\n{text_content}\n</user_message>\n"
        "Respond to the user's message. Use the document context only if it's relevant to what they're asking "
        "while citing the sources using their titles. If they're just chatting or the context isn't helpful, "
        "respond naturally without forcing references to the documents."
    )


def make_docs_query(text_content: str, documents: list) -> str:
    sources = "\n".join(f"\n--- Source {i} ---\n{doc}." for i, doc in enumerate(documents))
    return clean_text(
        f"Based on the following sources, answer this question:\n\nQuestion: {text_content}\n\nSources:{sources}\n\nAnswer based on the sources above:"
    )


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[.\s\u00A0\u2009\u200B\u2024\u2027]{3,}", " ", text)

    def is_safe(c):
        return unicodedata.category(c)[0] in "LNZ" or c in "\n\t @#/\\_-+=*&%$:.?!"

    text = "".join(c if is_safe(c) else " " for c in text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\b.{1,4}?\b)(?:\s*\1){2,}", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text.strip()


# ── File readers ───────────────────────────────────────────────────────────────

def _append_hidden_links(text: str, links: list) -> str:
    links = [l for l in links if l]
    if not links:
        return text
    text_lower = text.lower()
    hidden = [l for l in links if l.replace("https://", "").replace("http://", "").replace("www.", "").lower() not in text_lower]
    if hidden:
        text += "\n\n=== Links ===\n" + "\n".join(hidden)
    return text


def _read_txt(file) -> str:
    return file.read().decode("utf-8")


def _read_markdown(file) -> str:
    return file.read().decode("utf-8")


def _read_html(file) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(file.read().decode("utf-8"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        return f"Error: {e}"


def _read_pdf(file) -> str:
    try:
        import fitz
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join(page.get_text("text", sort=True) for page in pdf)
        links = [link["uri"] for page in pdf for link in page.get_links() if "uri" in link]
        pdf.close()
        return _append_hidden_links(text, links)
    except Exception as e:
        return f"Error: {e}"


def _read_docx(file) -> str:
    try:
        import docx
        doc = docx.Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
        links = [rel.target_ref for rel in doc.part.rels.values() if "hyperlink" in rel.reltype]
        return _append_hidden_links(text, links)
    except Exception as e:
        return f"Error: {e}"


def _read_excel(file) -> str:
    try:
        import pandas as pd
        excel_file = pd.ExcelFile(file)
        return "\n\n".join(
            f"=== Sheet: {name} ===\n{pd.read_excel(file, sheet_name=name).to_string(index=False)}"
            for name in excel_file.sheet_names
        )
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        return ""


def _read_csv(file) -> str:
    try:
        import pandas as pd
        return pd.read_csv(file).to_string(index=False)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return ""


def _read_pptx(file) -> str:
    try:
        from pptx import Presentation
        prs = Presentation(file)
        text, links = "", []
        for i, slide in enumerate(prs.slides, 1):
            text += f"\n=== Slide {i} ===\n"
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
                if hasattr(shape, "click_action") and shape.click_action.hyperlink:
                    links.append(shape.click_action.hyperlink.address)
                if hasattr(shape, "text_frame"):
                    for para in shape.text_frame.paragraphs:
                        for run in para.runs:
                            if run.hyperlink and run.hyperlink.address:
                                links.append(run.hyperlink.address)
        return _append_hidden_links(text, links)
    except Exception as e:
        return f"Error: {e}"


def _read_json(file) -> str:
    try:
        return json.dumps(json.load(file), indent=2)
    except Exception as e:
        return f"Error: {e}"


def _read_rtf(file) -> str:
    try:
        try:
            import pypandoc
            with tempfile.NamedTemporaryFile(delete=False, suffix=".rtf") as f:
                f.write(file.read())
                tmp = f.name
            try:
                return pypandoc.convert_file(tmp, "plain", format="rtf")
            finally:
                os.unlink(tmp)
        except ImportError:
            from striprtf.striprtf import rtf_to_text
            file.seek(0)
            return rtf_to_text(file.read().decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


_READERS = {
    "txt": _read_txt, "md": _read_markdown, "html": _read_html,
    "pdf": _read_pdf, "docx": _read_docx, "xlsx": _read_excel,
    "csv": _read_csv, "pptx": _read_pptx, "json": _read_json, "rtf": _read_rtf,
}


def extract_text_from_file(file) -> str:
    key = _SUPPORTED_TYPES.get(file.type) or _EXT_MAP.get(os.path.splitext(file.name)[1].lower())
    reader = _READERS.get(key)
    return clean_text(reader(file)) if reader else ""
