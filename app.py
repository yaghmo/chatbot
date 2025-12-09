import streamlit as st
import json
import uuid
from datetime import datetime
from utils.api_client import APIClient


class SidebarManager:
    def __init__(self, cfg_file):
        if "sidebar_open" not in st.session_state:
            st.session_state.sidebar_open = True
        if "recent_chats" not in st.session_state:
            st.session_state.recent_chats = []
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None

        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
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

    def render(self):
        if st.session_state.sidebar_open:
            with st.sidebar:
                self._render_settings()
                self._render_new_chat_button()
                self._render_recent_chats()

    def _render_settings(self):
        with st.popover("⚙️", use_container_width=False):
            st.markdown("#### Settings")

            model_names = list(st.session_state.model_cfg.keys())

            default_index = 0
            if st.session_state.model_name in model_names:
                default_index = model_names.index(st.session_state.model_name)

            selected_name = st.selectbox(
                "Choose your model:",
                model_names,
                index=default_index,
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
                "Temperature", 0.0, 2.0, st.session_state.temperature, step=0.1
            )
            st.session_state.top_p = st.slider(
                "Top P", 0.0, 1.0, st.session_state.top_p, step=0.01
            )
            st.session_state.max_tokens = st.slider(
                "Max tokens",
                128,
                st.session_state.context_length,
                st.session_state.max_tokens,
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
                    f"💬 {chat['title'][:30]}",
                    key=f"chat_{chat['id']}",
                    use_container_width=True
                ):
                    self.load_conversation(chat['id'])

    def start_new_conversation(self):
        if len(st.session_state.messages) > 1:
            if st.session_state.current_chat_id:
                self.update_current_chat()
            else:
                self.save_to_recents()

        st.session_state.messages = [
            {"role": "assistant", "content": "Let's start chatting! 👇"}
        ]
        st.session_state.current_chat_id = None
        st.rerun()

    def save_to_recents(self):
        if len(st.session_state.messages) > 1:
            title = "New Chat"
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    title = msg["content"][:50]
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

    def render(self):
        self._render_header()
        self._render_chat_history()
        self._render_chat_input()

    def _text_formatter(self,content):
        ...

    def _render_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _render_chat_input(self):
        if prompt := st.chat_input("What is up?"):
            self._handle_user_message(prompt)

    def _handle_user_message(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.current_chat_id:
            self._update_existing_chat()
        else:
            self._create_new_chat()
\
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = self._generate_response()

        st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.current_chat_id:
            self._update_existing_chat()

    def _create_new_chat(self):
        title = "New Chat"
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                title = msg["content"][:50]
                break

        chat_id = str(uuid.uuid4())
        chat_data = {
            "id": chat_id,
            "title": title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": st.session_state.messages.copy()
        }

        st.session_state.recent_chats.insert(0, chat_data)
        st.session_state.current_chat_id = chat_id

    def _update_existing_chat(self):
        for chat in st.session_state.recent_chats:
            if chat["id"] == st.session_state.current_chat_id:
                chat["messages"] = st.session_state.messages.copy()
                chat["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                break

    def _generate_response(self):
        if not st.session_state.active_model_name:
            st.error("Please select and load a model first!")
            return "Model not loaded."

        message_placeholder = st.empty()
        full_response = ""
        
        prompt = self._build_prompt()

        max_tokens = st.session_state.max_tokens
        temperature = st.session_state.temperature
        top_p = st.session_state.top_p

        for token in APIClient.generate_stream(
            model_name=st.session_state.active_model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        ):
            full_response += token
            message_placeholder.markdown(full_response)
        
        # Remove cursor and show final response
        message_placeholder.markdown(full_response)
        return full_response

    def _build_prompt(self):
        parts = []
        parts.append(f"[INST] <<SYS>>{st.session_state.sysprompt} <</SYS>>[/INST]")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                parts.append(f"[INST] {msg['content']} [/INST]")
            elif msg["role"] == "assistant":
                parts.append(msg['content'])
        return " ".join(parts)

class ChatbotApp:
    def __init__(self, cfg_file: str = "utils/model_cfg.json"):
        self.sidebar = SidebarManager(cfg_file)
        self.chat = ChatInterface()

    def run(self):
        st.set_page_config(
            page_title="Yaghmood's Chatbot",
            page_icon="💬",
            layout="wide"
        )

        self.sidebar.render()
        self.chat.render()

        # # Debug info
        # st.markdown(
        #     f"Debug: temp={st.session_state.temperature}, "
        #     f"top_p={st.session_state.top_p}, "
        #     f"max_tokens={st.session_state.max_tokens}, "
        #     f"model={st.session_state.active_model_name}"
        # )

if __name__ == "__main__":
    app = ChatbotApp()
    app.run()