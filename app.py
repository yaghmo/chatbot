from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime

import streamlit as st

import utils.template_media as tm
from utils.api_client import APIClient

st.markdown(
    """
<style>
.file-card {
    background: rgba(255,255,255,0.05);
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.10);
}
.file-name {
    font-size: 15px;
    font-weight: 500;
}
</style>
""",
    unsafe_allow_html=True,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SidebarManager:
    def __init__(self, cfg_file):
        if "sidebar_open" not in st.session_state:
            st.session_state.sidebar_open = True
        if "recent_chats" not in st.session_state:
            st.session_state.recent_chats = []
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None
        if "current_messages" not in st.session_state:
            st.session_state.current_messages = []

        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.5
        if "top_p" not in st.session_state:
            st.session_state.top_p = 1.0
        if "max_tokens" not in st.session_state:
            st.session_state.max_tokens = 1024
        if "threshold" not in st.session_state:
            st.session_state.threshold = 1.0

        if "model_cfg" not in st.session_state:
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    st.session_state.model_cfg = json.load(f)
            except json.JSONDecodeError as e:
                st.error("Failed to load JSON", e)
            except FileNotFoundError as e:
                st.error("File not found!", e)
            if not isinstance(st.session_state.model_cfg, dict):
                raise ValueError("JSON root is not a dict")

        if "model_name" not in st.session_state:
            st.session_state.model_name = None
        if "active_model_name" not in st.session_state:
            st.session_state.active_model_name = None

        if "context_length" not in st.session_state:
            st.session_state.context_length = None

        if "sysprompt" not in st.session_state:
            try:
                with open(
                    os.path.join(os.getenv("CONFIG_DIR", "config"), "system_prompt.txt"), "r", encoding="utf-8"
                ) as f:
                    st.session_state.sysprompt = f.read()
            except FileNotFoundError as e:
                st.error("File not found!", e)

        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False

        if "list_of_path" not in st.session_state:
            st.session_state.list_of_path = []
        if "rag_docs" not in st.session_state:
            st.session_state.rag_docs = []
        if "collection" not in st.session_state:
            st.session_state.collection = None

    def render(self):
        if st.session_state.sidebar_open:
            with st.sidebar:
                self._render_settings()
                self._render_new_chat_button()
                st.divider()
                # self._render_blobal_RAG()
                # st.divider()
                self._render_recent_chats()

    def _render_settings(self):
        with st.popover("", use_container_width=False, disabled=st.session_state.is_generating, icon="⚙️"):
            st.markdown("#### Settings")

            model_names = list(st.session_state.model_cfg.keys())

            default_index = 0
            if st.session_state.model_name in model_names:
                default_index = model_names.index(st.session_state.model_name)

            selected_name = st.selectbox(
                "Choose your model:", model_names, index=default_index, disabled=st.session_state.is_generating
            )

            # Check if model changed
            model_changed = st.session_state.model_name != selected_name
            st.session_state.model_name = selected_name

            cfg = st.session_state.model_cfg[selected_name]
            st.caption(cfg["purpose"])

            st.session_state.context_length = cfg["context_length"]

            # Load model via API if changed
            if model_changed or st.session_state.active_model_name != selected_name:
                with st.spinner(f"Loading {selected_name}"):
                    result = APIClient.load_model(selected_name)
                    if result:
                        st.session_state.active_model_name = selected_name
                        st.success(f"✅ {selected_name} loaded successfully")

            # Parameter sliders
            st.session_state.temperature = st.slider(
                "Temperature",
                0.0,
                2.0,
                st.session_state.temperature,
                step=0.1,
                disabled=st.session_state.is_generating,
                help="Controls how creative or deterministic the model is. Lower = focused, higher = more diverse.",
            )
            st.session_state.top_p = st.slider(
                "Top-p",
                0.0,
                1.0,
                st.session_state.top_p,
                step=0.01,
                disabled=st.session_state.is_generating,
                help="Limits choices to the most probable words whose combined probability is p. Lower = safer, higher = more varied.",
            )
            st.session_state.max_tokens = st.slider(
                "Max tokens",
                128,
                st.session_state.context_length,
                st.session_state.max_tokens,
                disabled=st.session_state.is_generating,
                help="Sets the maximum length of the model’s response.",
            )
            st.session_state.threshold = st.slider(
                "Threshold",
                0.7,
                2.0,
                st.session_state.threshold,
                step=0.1,
                disabled=st.session_state.is_generating,
                help="RAG Similarity Threshold (0–2): Determines how closely a document must match your query to be included. Higher = more documents (looser matching), lower = fewer but more relevant documents (stricter matching).",
            )

    def _render_new_chat_button(self):
        if st.button("### New Chat", icon="➕"):
            self.start_new_conversation()

    def _render_blobal_RAG(self):
        """Render RAG document upload section"""
        if not st.toggle(
            "Activate RAG over whole account", key="rag_active", value=st.session_state.get("rag_active", False)
        ):
            return

    def _render_recent_chats(self):
        st.markdown("### Recent Chats")
        if not st.session_state.recent_chats:
            st.caption("No recent chats yet")
        else:
            for chat in st.session_state.recent_chats[:5]:
                if st.button(f"💬 {chat['title']}", key=f"chat_{chat['id']}", use_container_width=True):
                    self.load_conversation(chat["id"])

    def start_new_conversation(self):
        if st.session_state.current_chat_id and len(st.session_state.messages) > 1:
            self.update_current_chat()

        st.session_state.messages = [{"role": "assistant", "text": "Let's start chatting! 👇"}]
        st.session_state.current_chat_id = None
        st.rerun()

    def save_to_recents(self):
        mode = st.session_state.model_cfg[st.session_state.active_model_name].get("mode")

        if len(st.session_state.messages) > 1:
            for msg in st.session_state.messages:
                title = "New Chat"
                if msg["role"] == "user":
                    title = msg["text"]
                    if len(msg["text"].split()) > 10:
                        title = APIClient.generate(
                            model_name=st.session_state.active_model_name,
                            template=[
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": tm.title_prompt(msg["text"])}]
                                    if mode == "vlm"
                                    else tm.title_prompt(msg["text"]),
                                }
                            ],
                            max_tokens=8,
                        )
                    break

            chat_id = str(uuid.uuid4())
            chat_data = {
                "id": chat_id,
                "title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.messages.copy(),
            }
            st.session_state.recent_chats.insert(0, chat_data)
            st.session_state.current_chat_id = chat_id

    def update_current_chat(self):
        if st.session_state.current_chat_id:
            for chat in st.session_state.recent_chats:
                if chat["id"] == st.session_state.current_chat_id:
                    chat["messages"] = st.session_state.messages.copy()
                    chat["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    break

    def load_conversation(self, chat_id):
        for chat in st.session_state.recent_chats:
            if chat["id"] == chat_id:
                st.session_state.messages = chat["messages"].copy()
                st.session_state.current_chat_id = chat_id
                st.rerun()
                break


class ChatInterface:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "text": "What can I help with?"}]

    def _render_header(self):
        st.markdown(
            """
            <h1 style='text-align: center;'>💬 Yaghmo's chatbot</h1>
            <p style='text-align: center; color: gray;'>
                Note that this demo app is actually connected to a Language Model.
                Choose which one you would want to use in the settings.
            </p>
            """,
            unsafe_allow_html=True,
        )

    def _render_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message.get("role", "assistant")):
                medias = message.get("medias", [])
                docs = message.get("docs", [])

                if medias or docs:
                    self._display_files_as_cards(medias=medias, docs=docs)

                if message.get("audio"):
                    st.audio(message["audio"])

                st.markdown(message.get("text", ""))

    def _save_current_chat_to_recents(self):
        """Save or update current chat in recents list"""

        if not st.session_state.current_chat_id:
            return

        # Check if chat already exists in recents
        existing_chat = None
        for chat in st.session_state.recent_chats:
            if chat["id"] == st.session_state.current_chat_id:
                existing_chat = chat
                break

        if existing_chat:
            # Update existing chat
            existing_chat["messages"] = st.session_state.messages.copy()
            existing_chat["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        else:
            # Create new chat in recents
            mode = st.session_state.model_cfg[st.session_state.active_model_name].get("mode")

            # Generate title from first user message
            title = "New Chat"
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    title = msg["text"]
                    if len(msg["text"].split()) > 10:
                        title = APIClient.generate(
                            model_name=st.session_state.active_model_name,
                            template=[
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": tm.title_prompt(msg["text"])}]
                                    if mode == "vlm"
                                    else tm.title_prompt(msg["text"]),
                                }
                            ],
                            max_tokens=8,
                        )
                    break

            chat_data = {
                "id": st.session_state.current_chat_id,
                "title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.messages.copy(),
            }
            st.session_state.recent_chats.insert(0, chat_data)

    def _render_chat_input(self):
        ALLOWED_EXTENSIONS = [
            # images
            "png",
            "jpg",
            "jpeg",
            "gif",
            "bmp",
            "webp",
            "tiff",
            # videos
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
            # documents
            "txt",
            "md",
            "pdf",
            "docx",
            "doc",
            "xlsx",
            "xls",
            "csv",
            "pptx",
            "ppt",
            "json",
            "html",
            "htm",
            "rtf",
        ]
        if prompt := st.chat_input(
            "Ask anything",
            accept_audio=True,
            accept_file="multiple",
            file_type=ALLOWED_EXTENSIONS,
        ):
            self._handle_user_message(prompt)

    def _display_files_as_cards(self, medias=None, docs=None, max_preview=3):
        main_col, right_col = st.columns([5, 2])

        with right_col:
            preview_medias = medias[:max_preview]
            extra_medias = medias[max_preview:]

            if preview_medias:
                cols = st.columns(len(preview_medias), gap="small")
                for col, media in zip(cols, preview_medias):
                    with col:
                        media_type = media.type.split("/")[0]
                        if media_type == "image":
                            st.image(media, width="stretch")
                        elif media_type == "video":
                            st.video(media)
                        else:
                            st.markdown(f"📄 {media.name}")

            if extra_medias:
                with st.expander(f"+{len(extra_medias)} more media"):
                    for media in extra_medias:
                        st.markdown(f"**{media.name}**")
                        media_type = media.type.split("/")[0]
                        if media_type == "image":
                            st.image(media, width="stretch")
                        elif media_type == "video":
                            st.video(media)

            if docs:
                if preview_medias:
                    st.divider()
                st.caption("Documents")
                for file in docs:
                    st.markdown(f"📄 {file.rsplit('_', 2)[0]}")

    def _handle_user_message(self, prompt):

        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid.uuid4())

        mode = st.session_state.model_cfg[st.session_state.active_model_name].get("mode")
        model_name = st.session_state.active_model_name
        text_content = prompt.text or ""
        media_files = []
        doc_files = []
        prompt_docs = []
        vlm_content = []
        text_doc = ""
        message_placeholder = st.empty()

        if prompt.audio:
            message_placeholder = st.markdown("*Processing audio...*")
            suffix = os.path.splitext(prompt.audio.name)[1] or ".wav"
            prompt.audio.seek(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(prompt.audio.read())
                temp_path = f.name
            text_content = APIClient.transcribe(audio_path=temp_path)
            st.session_state.list_of_path.append(temp_path)
            message_placeholder.empty()

        content = text_content
        if prompt.files:
            media_files = [f for f in prompt.files if f.type and f.type.startswith(("image", "video"))]
            doc_files = [f for f in prompt.files if f not in media_files]
            if media_files:
                if mode == "vlm":
                    message_placeholder.markdown("*Analysing medias...*")
                    for file in media_files:
                        temp_path, media_type = tm.media_resize(file=file)
                        vlm_content.append({"type": media_type, media_type: temp_path})
                        st.session_state.list_of_path.append(temp_path)
                else:
                    st.toast(
                        f"⚠️ {len(media_files)} media file(s) detected but current model doesn't support vision. Please select a VLM for such feature.",
                        duration="long",
                    )
            message_placeholder.empty()
            if doc_files:
                message_placeholder.markdown("*Analysing documents...*")
                for file in doc_files:
                    text_doc = tm.extract_text_from_file(file)
                    all_doc = list(set(prompt_docs + st.session_state.rag_docs))
                    is_sim = APIClient.check_sim(
                        model_name=st.session_state.active_model_name, user="user", all_doc=all_doc, text_doc=text_doc
                    )

                    if text_doc and not is_sim:
                        introduction = text_doc[:3000]
                        summary = APIClient.generate(
                            model_name=st.session_state.active_model_name,
                            template=[
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": tm.summary_prompt(introduction)}]
                                    if mode == "vlm"
                                    else tm.summary_prompt(introduction),
                                }
                            ],
                            max_tokens=300,
                        )

                        doc_id = APIClient.add_document(
                            user="user",
                            chat_id=st.session_state.current_chat_id,
                            file_name=file.name,
                            summary=summary,
                            content=text_doc,
                            model_name=st.session_state.active_model_name,
                        )
                        prompt_docs.append(doc_id)
                    elif not text_doc:
                        st.toast("⚠️ Some documents contain no text (scans, empty, etc.)")
                if len(prompt_docs) != len(doc_files):
                    st.toast("⚠️ Some documents are no saved for being identical with some others.")

        st.session_state.rag_docs.extend(prompt_docs)
        if st.session_state.rag_docs:
            relevent_chunks = APIClient.rag_query(
                user="user",
                chat_docs=st.session_state.rag_docs,
                query=text_content,
                top_n=5,
                threshold=0.17,
            )
            logger.info(f"Relevent chunks : {relevent_chunks}")

            if len(relevent_chunks) > 0:
                content = tm.make_rag_query(text_content, relevent_chunks)

        message_placeholder.empty()

        vlm_content.append({"type": "text", "text": content})

        # check for history
        st.session_state.current_messages = tm.summarize_history(
            messages=st.session_state.current_messages,
            mode=mode,
            model_name=model_name,
            max_window_size=int(st.session_state.context_length * 0.69),
        )

        message = {
            "role": "user",
            "text": text_content,
            "content": content,
            "vlm_content": vlm_content,
            "medias": media_files,
            "audio": prompt.audio or None,
            "docs": prompt_docs,
        }

        st.session_state.messages.append(message)
        st.session_state.current_messages.append(message)

        with st.chat_message("user"):
            if any(file in message for file in ("medias", "docs")):
                self._display_files_as_cards(medias=message["medias"], docs=message["docs"])
            if prompt.audio:
                st.audio(prompt.audio)
                st.markdown(text_content)
            else:
                st.markdown(text_content)
        self._save_current_chat_to_recents()

        with st.chat_message("assistant"):
            template = tm.build_prompt_template(
                messages=st.session_state.current_messages, mode=mode, system_prompt=st.session_state.sysprompt
            )
            response = self._generate_response(template)
            # response = content
            tm.clear_temp(list_of_path=st.session_state.list_of_path)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "vlm_content": [{"type": "text", "text": response}],
                "content": response,
                "text": response,
            }
        )

        st.session_state.current_messages.append(st.session_state.messages[-1])
        self._save_current_chat_to_recents()
        st.rerun()

    def _generate_response(self, template):
        if not st.session_state.active_model_name:
            st.error("Please select and load a model first!")
            return "Model not loaded."

        message_placeholder = st.empty()
        message_placeholder.markdown("*Thinking...*")
        full_response = ""
        max_tokens = st.session_state.max_tokens
        temperature = st.session_state.temperature
        top_p = st.session_state.top_p

        st.session_state.is_generating = True

        try:
            for token in APIClient.generate_stream(
                model_name=st.session_state.active_model_name,
                template=template,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                full_response += token
                message_placeholder.markdown(full_response)

            message_placeholder.markdown(full_response)
            return full_response

        finally:
            st.session_state.is_generating = False

    def render(self):
        self._render_header()
        self._render_chat_history()
        self._render_chat_input()


class ChatbotApp:
    def __init__(self, cfg_file: str = os.getenv("MODEL_CONFIG", os.path.join("config", "model_cfg.json"))):
        self.sidebar = SidebarManager(cfg_file)
        self.chat = ChatInterface()

    def run(self):
        st.set_page_config(page_title="Yaghmood's Chatbot", page_icon="💬", layout="wide")

        self.chat.render()
        self.sidebar.render()


if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
