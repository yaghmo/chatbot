import streamlit as st
import json
import uuid
from datetime import datetime
from utils.api_client import APIClient
import utils.template_media as tm
import os
import time

st.markdown("""
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
""", unsafe_allow_html=True)



class SidebarManager:
    def __init__(self, cfg_file):
        if "sidebar_open" not in st.session_state:
            st.session_state.sidebar_open = True
        if "recent_chats" not in st.session_state:
            st.session_state.recent_chats = []
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None

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
                with open(os.path.join("config","system_prompt.txt"), "r", encoding="utf-8") as f:
                    st.session_state.sysprompt = f.read()
            except FileNotFoundError as e:
                st.error("File not found!", e)

        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False

        if "list_of_path" not in st.session_state:
            st.session_state.list_of_path = []
        if "rag_files" not in st.session_state:
            st.session_state.rag_files = []
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        if "rag_active" not in st.session_state:
            st.session_state.rag_active = False
        if "rag_active2" not in st.session_state:
            st.session_state.rag_active2 = False
        if "collection" not in st.session_state:
            st.session_state.collection = None

    def render(self):
        if st.session_state.sidebar_open:
            with st.sidebar:
                self._render_settings()
                self._render_new_chat_button()
                st.divider()
                self._render_RAG_upload()
                st.divider()
                self._render_recent_chats()

    def _render_settings(self):
        with st.popover("", use_container_width=False, disabled=st.session_state.is_generating,icon="⚙️"):
            st.markdown("#### Settings")

            model_names = list(st.session_state.model_cfg.keys())

            default_index = 0
            if st.session_state.model_name in model_names:
                default_index = model_names.index(st.session_state.model_name)

            selected_name = st.selectbox(
                "Choose your model:",
                model_names,
                index=default_index,
                disabled=st.session_state.is_generating
            )

            # Check if model changed
            model_changed = (st.session_state.model_name != selected_name)
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
                "Temperature", 0.0, 2.0, st.session_state.temperature, step=0.1, disabled=st.session_state.is_generating,
                help = "Controls how creative or deterministic the model is. Lower = focused, higher = more diverse."
            )
            st.session_state.top_p = st.slider(
                "Top-p", 0.0, 1.0, st.session_state.top_p, step=0.01, disabled=st.session_state.is_generating,
                help="Limits choices to the most probable words whose combined probability is p. Lower = safer, higher = more varied."
            )
            st.session_state.max_tokens = st.slider(
                "Max tokens",
                128,
                st.session_state.context_length,
                st.session_state.max_tokens,
                disabled=st.session_state.is_generating,
                help="Sets the maximum length of the model’s response."
            )
            st.session_state.threshold = st.slider(
                "Threshold",
                0.7,
                2.0,
                st.session_state.threshold,
                step=0.1,
                disabled=st.session_state.is_generating,
                help="RAG Similarity Threshold (0–2): Determines how closely a document must match your query to be included. Higher = more documents (looser matching), lower = fewer but more relevant documents (stricter matching)."
            )

    def _render_new_chat_button(self):
        if st.button("### New Chat",icon="➕"):
            self.start_new_conversation()
    
    def _render_RAG_upload(self):
        """Render RAG document upload section"""

        # Constants
        ALLOWED_EXTENSIONS = ['txt', 'md', 'pdf', 'docx', 'doc', 'xlsx', 'xls', 
                            'csv', 'pptx', 'ppt', 'json', 'html', 'htm', 'rtf']
        MAX_SOURCES = 10

        st.markdown("### Sources")

        if not st.toggle(
            "Activate RAG",
            key="rag_active",  
            value=st.session_state.get("rag_active", False)
        ):
            return
        
        # Show current sources count
        current_count = len(st.session_state.rag_files)
        remaining = MAX_SOURCES - current_count
        
        if remaining > 0:
            # File uploader
            uploaded_files = st.file_uploader(
                label=f"Add sources ({current_count}/{MAX_SOURCES})",
                accept_multiple_files=True,
                type=ALLOWED_EXTENSIONS,
                key=f"rag_uploader_{st.session_state.uploader_key}",
                help=f"You can upload up to {remaining} more document(s) - OCR feature is not implemented."
            )
            
            if uploaded_files:
                # Check if adding would exceed limit
                new_count = current_count + len(uploaded_files)
                
                
                if new_count > MAX_SOURCES:
                    st.warning(f"⚠️ Can only add {remaining} more document(s). Please select fewer files.")
                else:
                    # Add button
                    if st.button("Add Sources"):
                        for file in uploaded_files:
                            text_doc = tm.extract_text_from_file(file)
                            if text_doc:
                                if not st.session_state.rag_active2:
                                    st.session_state.rag_active2 = True
                                    st.session_state.collection = tm.init_chromadb()
                                with st.spinner("Processing documents..."): 
                                    summary = tm.summary_prompt(text_doc[:1500])
                                    summary = APIClient.generate(
                                        model_name=st.session_state.active_model_name,
                                        template=[{"role": "user", "content":[{"type":"text", "text":summary}] if mode == "vlm" else summary}],
                                        max_tokens=300,
                                    )
                                    st.session_state.rag_files.append(file.name)
                                    tm.add_document_to_chromadb(st.session_state.collection,file.name,text_doc,summary)
                            else:
                                st.warning(f"⚠️ Some documents contain no text (scans, empty, etc.)")
                                time.sleep(4)
                                    
                        # Reset uploader
                        st.session_state.uploader_key += 1
                        st.rerun()
        else:
            st.info(f"Maximum sources reached ({MAX_SOURCES}/{MAX_SOURCES})")
        
        # Display current sources
        if st.session_state.rag_files:
            st.divider()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(f"**{len(st.session_state.rag_files)} source(s) loaded**")
            with col2:
                if st.button("Clear all", key="clear_rag"):
                    st.session_state.rag_files = []
                    st.session_state.collection.delete(ids=st.session_state.collection.get()["ids"])
                    st.rerun()

            with st.expander("View sources", expanded=False):
                for idx, filename in enumerate(st.session_state.rag_files):
                    col_name, col_btn = st.columns([0.85, 0.15])

                    with col_name:
                        st.markdown(
                            f"<div class='file-card'><span class='file-name'>{idx}. {filename}</span></div>",
                            unsafe_allow_html=True
                        )

                    with col_btn:
                        if st.button("✖", key=f"remove_{idx}", help=f"Remove {filename}", use_container_width=True):
                            st.session_state.rag_files.pop(filename)  
                            st.rerun()

    def _render_recent_chats(self):
        st.markdown("### Recent Chats")
        if not st.session_state.recent_chats:
            st.caption("No recent chats yet")
        else:
            for chat in st.session_state.recent_chats[:5]:
                if st.button(
                    f"💬 {chat['title']}",
                    key=f"chat_{chat['id']}",
                    use_container_width=True
                ):
                    self.load_conversation(chat['id'])

    def start_new_conversation(self):
        if st.session_state.current_chat_id and len(st.session_state.messages) > 1:
            self.update_current_chat()
        
        st.session_state.messages = [
            {"role": "assistant", "text": "Let's start chatting! 👇"}
        ]
        st.session_state.current_chat_id = None
        st.rerun()

    def save_to_recents(self):
        mode = st.session_state.model_cfg[st.session_state.active_model_name].get("mode")
        
        if len(st.session_state.messages) > 1:
            for msg in st.session_state.messages:
                title = "New chat"
                if msg["role"] == "user":
                    if len(msg["text"])>0:
                        title = APIClient.generate(
                        model_name=st.session_state.active_model_name,
                        template=[{"role": "user", "content":[{"type":"text", "text":tm.system_instruction(msg["text"])}] if mode == "vlm" else tm.system_instruction(msg["text"])}]
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
            st.session_state.messages = [
                {"role": "assistant", "text": "What can I help with?"}
            ]

    def _render_header(self):
        st.markdown(
            """
            <h1 style='text-align: center;'>💬 Yaghmo's chatbot</h1>
            <p style='text-align: center; color: gray;'>
                Note that this demo app is actually connected to a Language Model. 
                Choose which one you would want to use in the settings.
            </p>
            """,
            unsafe_allow_html=True
        )


    def _render_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "files" in message:
                    self._display_files_as_cards(message["files"])
                st.markdown(message.get("text",""))


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
                title = "New chat"
                if msg["role"] == "user":
                    if len(msg["text"])>0:
                        title = APIClient.generate(
                            model_name=st.session_state.active_model_name,
                            template=[{
                                "role": "user", 
                                "content": [{"type":"text", "text":tm.system_instruction(msg["text"])}] if mode == "vlm" else tm.system_instruction(msg["text"])
                            }]
                        )
                    break
            
            chat_data = {
                "id": st.session_state.current_chat_id,
                "title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.messages.copy()
            }
            st.session_state.recent_chats.insert(0, chat_data)

    def _render_chat_input(self):
        ALLOWED_EXTENSIONS = [
        # images
        "png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff",
        # videos
        "mp4", "mov", "avi", "mkv", "webm",
        # documents
        "pdf", "txt", "csv", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "json"
        ]
        if prompt := st.chat_input(
            "Ask anything",
            accept_audio=True,
            accept_file="multiple",
            file_type=ALLOWED_EXTENSIONS,
            ):
            self._handle_user_message(prompt)

    def _display_files_as_cards(self, files, max_preview=3):
        """
        Display uploaded files as small cards on the right side
        with optional "+N more" button above the user input area.
        """
        main_col, right_col = st.columns([5, 2])

        with right_col:
            preview_files = files[:max_preview]
            extra_files = files[max_preview:]

            if preview_files:
                cols = st.columns(len(preview_files), gap="small")
                for col, file in zip(cols, preview_files):
                    with col:
                        file_type = file.type.split("/")[0]
                        if file_type == "image":
                            st.image(file, width="content")
                        elif file_type == "video":
                            st.video(file)
                        else:
                            st.markdown(f"📄 {file.name}")

            if extra_files:
                with st.expander(f"+{len(extra_files)} more"):
                    for file in extra_files:
                        st.markdown(f"**{file.name}**")
                        file_type = file.type.split("/")[0]
                        if file_type == "image":
                            st.image(file, width="content")
                        elif file_type == "video":
                            st.video(file)
                        else:
                            st.markdown(f"📄 {file.name}")

    def _handle_user_message(self, prompt):
        
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid.uuid4())

        mode = st.session_state.model_cfg[st.session_state.active_model_name].get("mode")
        MAX_TOKENS = 640
        num_tokens = 0 
        text_content = prompt.text or ""
        files_info = []

        if prompt.audio:
            message_placeholder = st.markdown("Processing audio...")
            text_content = "AHOL"
            #voice recog
        
        content = text_content
        vlm_content = []
        documents = []
        text_doc = ""
        message_placeholder = st.empty()
        if prompt.files:
            media_files = [
                f for f in prompt.files
                if f.type and f.type.startswith(("image", "video"))
            ]
            doc_files = [f for f in prompt.files if f not in media_files]
            if media_files:
                if mode == "vlm":
                    message_placeholder.markdown("Analysing medias...")
                    for file in media_files:
                        temp_path, media_type = tm.media_resize(file=file)
                        vlm_content.append({
                            "type": media_type,
                            media_type: temp_path
                        })
                        st.session_state.list_of_path.append(temp_path)
                        files_info.append(file)
                else:
                    st.toast(f"⚠️ {len(media_files)} media file(s) detected but current model doesn't support vision. Please select a VLM for such feature.",duration = "long")

            if doc_files:
                rag_files = []
                message_placeholder.markdown("Analysing documents...")
                for doc in doc_files:
                    text_doc = tm.extract_text_from_file(doc)
                    num_tokens = APIClient.count_tokens(
                        model_name = st.session_state.active_model_name,
                        text = text_doc
                        )
                    st.write(text_doc[:])
                    if num_tokens < MAX_TOKENS:
                        documents.append(text_doc)
                        files_info.append(doc)
                if len(documents)<len(doc_files):
                    st.toast(f"⚠️ some documents are too large, please activate the RAG option (in settings) and upload them there.",duration = "long")
            
            if documents:
                content = tm.make_docs_query(text_content,documents)                         

        if st.session_state.rag_active and st.session_state.rag_files: 
            message_placeholder.markdown("Analysing documents...")
            relevent_chunks = tm.hierarchical_retrieval(collection = st.session_state.collection, old_files=None, chat_files=st.session_state.rag_files, query = content, message_placeholder=message_placeholder)
            if len(relevent_chunks)>0:
                content = tm.rag_prompt(text_content, relevent_chunks)
            # else:
            #     st.warning("No relevant information was found in the supplied RAG sources, Consider disabling RAG if you prefer a general response.")

            # st.write(relevent_docs)

        vlm_content.append({"type": "text", "text": content})

        message = {
        "role": "user",
        "text": text_content,
        "content": content,
        "vlm_content": vlm_content,
        "files": files_info
        }
        st.session_state.messages.append(message)

        with st.chat_message("user"):
            if "files" in message:
                self._display_files_as_cards(message["files"])
            st.markdown(text_content)

        self._save_current_chat_to_recents()

        with st.chat_message("assistant"):
            if message["text"] == "" and len(message["files"]) == 0:
                response = "Hum? you forgot to say something ?"
                st.markdown(response)
            else:
                template = tm.build_prompt_template(messages=st.session_state.messages.copy()[1:],system_prompt=st.session_state.sysprompt,mode=mode)
                response = self._generate_response(template)
                # response = "basdf"
                tm.clear_temp(list_of_path= st.session_state.list_of_path)

        st.session_state.messages.append({
            "role": "assistant",
            "vlm_content": [{"type": "text", "text": response}],
            "content": response,
            "text": response
        })
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
                top_p=top_p
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
    def __init__(self, cfg_file: str = os.path.join("config","model_cfg.json")):
        self.sidebar = SidebarManager(cfg_file)
        self.chat = ChatInterface()

    def run(self):
        st.set_page_config(
            page_title="Yaghmood's Chatbot",
            page_icon="💬",
            layout="wide"
        )

        self.chat.render()
        self.sidebar.render()

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()