import streamlit as st
import random
import time
from datetime import datetime
import json 
import uuid
import utils.model

class SidebarManager:
    
    def __init__(self,cfg_file):
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
            st.session_state.max_tokens = 0
        if "model_cfg" not in st.session_state:
            with open(cfg_file, "r", encoding="utf-8") as f:
                st.session_state.model_cfg = json.load(f)
        if "model_name" not in st.session_state:
            st.session_state.model_name = None
        if "model" not in st.session_state:
            st.session_state.model = None
    
    def render(self):
        if st.session_state.sidebar_open:
            with st.sidebar:
                self._redner_settins()
                self._render_new_chat_button()
                self._render_recent_chats()


    def _redner_settins(self):
        with st.popover(f"⚙️", use_container_width=False):
            st.markdown("#### Settings")
            st.session_state.model_name = st.selectbox("Choose your model:",list(st.session_state.model_cfg.keys()))
            st.caption(st.session_state.model_cfg[st.session_state.model_name]["purpose"])
            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 2.0, st.session_state.temperature, step=0.1
            )
            st.session_state.top_p = st.slider(
                "Top P", 0.0, 1.0, st.session_state.top_p
            ) 
            st.session_state.max_tokens = st.slider(
                "Max tokens", 256, st.session_state.model_cfg[st.session_state.model_name]["max_tokens"], 256
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
            for i, chat in enumerate(st.session_state.recent_chats[:5]):
                if st.button(
                    f"💬 {chat['title'][:30]}",
                    key=f"chat_{chat['id']}",
                    use_container_width=True
                ):
                    self.load_conversation(chat['id'])
    
    def start_new_conversation(self):
        # Save current conversation to recents if it has messages
        if len(st.session_state.messages) > 1:
            if st.session_state.current_chat_id:
                self.update_current_chat()
            else:
                self.save_to_recents()

        # Reset messages
        st.session_state.messages = [
            {"role": "assistant", "content": "Let's start chatting! 👇"}
        ]
        st.session_state.current_chat_id = None
        st.rerun()
    
    def save_to_recents(self):
        """Save current conversation to recents"""
        if len(st.session_state.messages) > 1:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    title = msg["content"][:20]
                    break
            
            chat_id = str(uuid.uuid4())
            chat_data = {
                "title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.messages.copy(),
            }
            st.session_state.recent_chats.insert(0, chat_data)
            st.session_state.current_chat_id = chat_id

    def update_current_chat(self):
        """Update the currently loaded chat in recents"""
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
                {"role": "assistant", "content": "Let's start chatting! 👇"}
            ]
    
    def render(self):
        self._render_header()
        self._render_chat_history()
        self._render_chat_input()
    
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
        """Display chat messages from history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def _render_chat_input(self):
        # col1, col2 = st.columns([10, 1])
        
        # with col1:
        #     if st.button("️+️", key="add_btn", help="Add attachment"):
        #         st.info("Attachment feature coming soon!")

        if prompt := st.chat_input("What is up?"):
            self._handle_user_message(prompt)
    
    def _handle_user_message(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.current_chat_id:
            self._update_existing_chat()
        else:
            self._create_new_chat()

        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = self._generate_response()
            self._display_streaming_response(response)
    
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        if st.session_state.current_chat_id:
            self._update_existing_chat()
        
        if len(st.session_state.messages) > 30:
             st.session_state.messages = st.session_state.messages[-30:]


    
    def _create_new_chat(self):
        """Create a new chat in recents"""
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
        """Update existing chat in recents"""
        for chat in st.session_state.recent_chats:
            if chat["id"] == st.session_state.current_chat_id:
                chat["messages"] = st.session_state.messages.copy()
                chat["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                break

    def _generate_response(self):
        return random.choice([
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ])
    
    def _build_prompt(self):
        parts = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                parts.append(f"[INST] {msg['content']} [/INST]")
            elif msg["role"] == "assistant":
                parts.append(msg['content'])
        return " ".join(parts)



    def _display_streaming_response(self, response):
        message_placeholder = st.empty()
        full_response = ""
        
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)


class ChatbotApp:
    def __init__(self,cfg_file:str="utils/model_cfg.json"):
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
        st.markdown(f"{st.session_state.top_p,st.session_state.temperature,st.session_state.model_name,st.session_state.current_chat_id}")
if __name__ == "__main__":
    app = ChatbotApp()
    app.run()