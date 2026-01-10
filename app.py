import streamlit as st
import json
import uuid
from datetime import datetime
from utils.api_client import APIClient
from PIL import Image
import os
import utils.template_media as tm
import tempfile
import time

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

        if "model_cfg" not in st.session_state:
            with open(cfg_file, "r", encoding="utf-8") as f:
                st.session_state.model_cfg = json.load(f)

        if "model_name" not in st.session_state:
            st.session_state.model_name = None
        if "active_model_name" not in st.session_state:
            st.session_state.active_model_name = None

        if "context_length" not in st.session_state:
            st.session_state.context_length = None

        if "sysprompt" not in st.session_state:
            with open("utils/system_prompt.txt", "r", encoding="utf-8") as f:
                st.session_state.sysprompt = f.read()

        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False  

    def render(self):
        if st.session_state.sidebar_open:
            with st.sidebar:
                self._render_settings()
                self._render_new_chat_button()
                self._render_recent_chats()

    def _render_settings(self):
        with st.popover("⚙️", use_container_width=False, disabled=st.session_state.is_generating):
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
                "Temperature", 0.0, 2.0, st.session_state.temperature, step=0.1, disabled=st.session_state.is_generating
            )
            st.session_state.top_p = st.slider(
                "Top P", 0.0, 1.0, st.session_state.top_p, step=0.01, disabled=st.session_state.is_generating
            )
            st.session_state.max_tokens = st.slider(
                "Max tokens",
                128,
                st.session_state.context_length,
                st.session_state.max_tokens,
                disabled=st.session_state.is_generating
            )

    def _render_new_chat_button(self):
        if st.button("### ➕ New Chat"):
            self.start_new_conversation()
        st.divider()

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
                if msg["role"] == "user":
                    title = APIClient.generate(
                    model_name=st.session_state.active_model_name,
                    template=[{"role": "user", "content":[{"type":"text", "text":tm.system_instruction(msg["content"])}] if mode == "vlm" else tm.system_instruction(msg["content"])}]
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
                {"role": "assistant", "content": "What can I help with?"}
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
                st.markdown(message.get("text", message.get("content", "")))


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
                    try:
                        title = APIClient.generate(
                            model_name=st.session_state.active_model_name,
                            template=[{
                                "role": "user", 
                                "content": [{"type":"text", "text":tm.system_instruction(msg["content"])}] if mode == "vlm" else tm.system_instruction(msg["content"])
                            }]
                        )
                        # Fallback if generation fails
                        if not title or title == "":
                            title = msg.get("text", "New Chat")[:50]
                    except Exception as e:
                        title = msg.get("text", "New Chat")[:50]
                    break
            
            chat_data = {
                "id": st.session_state.current_chat_id,
                "title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.messages.copy()
            }
            st.session_state.recent_chats.insert(0, chat_data)

    def _render_chat_input(self):
        if prompt := st.chat_input(
            "Ask anything",
            accept_audio=True,
            accept_file="multiple",
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
        files_info = []        
        processed = False
        text_content = ""

        if prompt.text:
            text_content = prompt.text
        
        if prompt.audio:
            text_content = "AHOL"
            #voice recog
        
        content = [{"type": "text", "text": text_content}]

        if prompt.files:
            for file in prompt.files:
                files_info.append(file)
            
            has_media = any(f.type.startswith(("image", "video")) for f in prompt.files)
            
            if has_media:
                if mode == "vlm":
                    content = []
                    for file in prompt.files:
                        temp_path = tm.media_resize(file=file)
                        if file.type.startswith("image"):
                            content.append({
                                "type": "image",
                                "image": temp_path
                            })
                        elif file.type.startswith("video"):
                            content.append({
                                "type": "video",
                                "video": temp_path
                            })
                        else:
                            # Non-media files in VLM context
                            st.info(f"📄 {file.name} ({file.type}) - not processed by VLM")
                            processed = True
                            # content = new content 
                    
                    if not processed:
                        if mode == "vlm":
                            content.append({"type": "text", "text": text_content})
                        else:
                            content = text_content
                else:
                    st.warning(f"⚠️ {len([f for f in prompt.files if f.type.startswith(('image', 'video'))])} media file(s) uploaded but current model doesn't support vision. Files are saved for later use.")
            else:
                # Only non-media files (documents, etc.)
                for file in prompt.files:
                    st.info(f"📄 Uploaded: {file.name} ({file.type})")
                # TODO: Process documents if needed
                content = "RAG"

        message = {
        "role": "user",
        "text": text_content,
        "content": content,
        "files": files_info
        }

        st.session_state.messages.append(message)

        with st.chat_message("user"):
            if "files" in message:
                self._display_files_as_cards(message["files"])
            st.markdown(text_content)

        self._save_current_chat_to_recents()

        with st.chat_message("assistant"):
            
            template = tm.build_prompt_template(messages=st.session_state.messages.copy()[1:],system_prompt=st.session_state.sysprompt,mode=mode)
            response = self._generate_response(template)

        st.session_state.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
            "text": response
        })

        self._save_current_chat_to_recents()


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
            first_token = True
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
    def __init__(self, cfg_file: str = "config/model_cfg.json"):
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