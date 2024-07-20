import streamlit as st
import os
import io
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit.components.v1 as components

# Constants
CHUNK_SIZE = 1024
XI_API_KEY = st.secrets["XI_API_KEY"]
VOICE_ID = st.secrets["VOICE_ID"]

# Load and process the PDF
@st.cache_resource
def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Create vector store
@st.cache_resource
def create_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Set up the conversational chain
@st.cache_resource
def setup_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return chain

# Chat function
def chat_with_pdf(chain, query, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']

# Text to speech function
def text_to_speech(text):
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    response = requests.post(tts_url, headers=headers, json=data, stream=True)
    
    if response.ok:
        audio_data = io.BytesIO()
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            audio_data.write(chunk)
        
        audio_data.seek(0)
        return audio_data
    else:
        st.error(f"TTS API request failed: {response.text}")
        return None

# Streamlit app
def main():
    st.title("Voice Chat with Your PDF")
    
    # File uploader
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf_file is not None:
        # Process the PDF
        with st.spinner("Processing PDF..."):
            texts = process_pdf(pdf_file)
            vector_store = create_vector_store(texts)
            chain = setup_chain(vector_store)
        
        st.success("PDF processed successfully!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Voice input using JavaScript
        st.write("Click the button and speak your question:")
        voice_input = st.empty()
        speech_recognition_script = """
        <script>
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById('voice-input').value = transcript;
            document.getElementById('submit-voice').click();
        };
        
        function startListening() {
            recognition.start();
        }
        </script>
        <button onclick="startListening()">Start Listening</button>
        <input type="text" id="voice-input" style="display:none;">
        <button id="submit-voice" style="display:none;"></button>
        """
        components.html(speech_recognition_script, height=100)
        
        if st.button("Submit Voice Input", key="submit-voice"):
            prompt = st.session_state.widget_values['voice-input']
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = chat_with_pdf(chain, prompt, [(msg["content"], "") for msg in st.session_state.messages if msg["role"] == "assistant"])
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Generate and play audio
                    audio_data = text_to_speech(response)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
                        
                        # Automatically play audio
                        autoplay_audio = f"""
                        <script>
                        const audio = new Audio('{audio_data}');
                        audio.play();
                        </script>
                        """
                        components.html(autoplay_audio, height=0)

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    main()