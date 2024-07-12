import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import sounddevice as sd
import scipy.io.wavfile as wavfile

load_dotenv()
total_data = []
file_path = 'U_audio.wav'

# Check if the file exists
if os.path.exists(file_path):
    # Delete the file
    os.remove(file_path)
def record_and_save_audio():
    sample_rate = 44100

    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'recording' not in st.session_state:
        st.session_state.recording = None

    if st.button("Start/Stop Recording", key="record_button"):
        if not st.session_state.is_recording:
            st.session_state.is_recording = True
            st.session_state.recording = sd.rec(int(sample_rate * 5), samplerate=sample_rate, channels=1, dtype='int16')  # Record for 5 seconds
            st.write("Recording audio...")
        else:
            st.session_state.is_recording = False
            sd.stop()
            st.session_state.recording = st.session_state.recording.reshape(-1)
            wavfile.write('u_audio.wav', sample_rate, st.session_state.recording)
            st.write("Recording complete.")
            st.success("Audio saved as 'u_audio.wav'.")

def main():
    st.title("Voice Recorder")

    # Audio file upload
    st.sidebar.title("Audio File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        # Save the uploaded file with a specific name u_audio.wav
        with open("u_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("File saved as u_audio.wav")

    record_and_save_audio()

    # Check which audio file to use
    if os.path.exists("u_audio.wav"):
      audio_file_path = "u_audio.wav"
    else :
       audio_file_path =  "doc_patient.mp3"

    API_KEY = os.getenv("DG_API_KEY")
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    # Generate button
    if st.button("Generate"):
        try:
            # Create a Deepgram client using the API key
            deepgram = DeepgramClient(API_KEY)

            with open(audio_file_path, "rb") as file:
                buffer_data = file.read()

            payload = {
                "buffer": buffer_data,
            }

            # Configure Deepgram options for audio analysis
            options = PrerecordedOptions(
                model="nova-2-general",
                smart_format=True
            )

            # Call the transcribe_file method with the text payload and options
            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

            # Print the response
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]

            st.title("Audio transcript")
            st.write(transcript)

            st.divider()

            # OpenAI integration for generating SOAP notes
            # model = ChatOpenAI(model="gpt-3.5-turbo") #openai
            model = ChatGroq(model="llama3-70b-8192")  #groq
            

            prompt_soap_note = ChatPromptTemplate.from_template("""
                You are an expert medical assistant trained to generate detailed SOAP notes based on patient information. Your task is to create a complete SOAP note that includes the following sections:

                "Here are the patient's data = {topic}
                you also need to figure out SOAP from the data."

                Subjective:
                - Summarize the patient's chief complaint, history of present illness, and any relevant past medical history, social history, or review of systems.

                Objective: 
                - Provide a summary of any relevant physical exam findings, vital signs, laboratory results, or other objective data collected during the patient encounter.

                Assessment:
                - Provide your assessment of the patient's condition, including any diagnoses or differential diagnoses.

                Plan:
                - Recommend an appropriate treatment plan, including any medications, therapies, referrals, or follow-up instructions.

                Use the following information about the patient to generate the SOAP note:
                
                Patient name
                Age
                History of present illness
                
                Respond with the complete SOAP note in the following format:

                Subjective:

                Objective: 

                Assessment:

                Plan:
                
            """)

            output_parser = StrOutputParser()

            chain_soap_note = prompt_soap_note | model | output_parser
            soap_note = chain_soap_note.invoke({"topic": transcript})
            
            st.title("SOAP note")
            soap_note_text = st.text_area("SOAP Note", value=soap_note, height=400)

            st.divider()

            # OpenAI integration for generating medication suggestions and patient advisories
            prompt_medications_and_advisories = ChatPromptTemplate.from_template("""
                You are an expert medical assistant trained to provide detailed recommendations for medications and patient advisories based on patient information. Your task is to generate a list of medications and advisories that includes the following sections:

                "Here are the patient's data = {topic}
                you also need to figure out appropriate medications and patient advisories from the data."

                Use the following information about the patient to generate the recommendations only if the patient provide this details :

                Patient name
                Age
                History of present illness
                
                Medications:
                - List of medications that the patient should take.

                Patient advisories:
                - Detailed list of do's and don'ts for the patient.

                Respond with the recommendations in the following format:
                
                Medications:
                    what madicions patient should take 

                Patient advisories:
                    what a patient should do 
                    
                    what a patient should not do 
         
            """)

            chain_medications_and_advisories = prompt_medications_and_advisories | model | output_parser
            medications_and_advisories = chain_medications_and_advisories.invoke({"topic": transcript})

            st.title("Medications and Patient Advisories")
            medications_and_advisories_text = st.text_area("Medications and Advisories", value=medications_and_advisories, height=400)

            # Combine all data and save to file
            total_data.append(transcript)
            total_data.append(soap_note)
            total_data.append(medications_and_advisories)
           
 
        except Exception as e:
            st.write(f"Exception: {e}")
            
    st.divider()
    
    st.header('Ask Anything about SOAP notes OR your health ')
    st.text('VKAPS It Solutions Pvt Ltd.')

    # Read the contents of the file
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # model = ChatOpenAI(model="gpt-3.5-turbo") #openai
    model = ChatGroq(model="llama3-70b-8192")  #groq
    output_parser = StrOutputParser()

    # Get user query input
    user_query = st.chat_input("Ask your query here about the given patient details...")

    # Create the chat prompt template
    totle_chat = ChatPromptTemplate.from_template(""" 
    You are a helpful AI assistant who is very humble and able to respond to every query of the user related to the users health.
    Provide accurate and straightforward responses to user queries. If the user greets you, respond with a greeting for example "Hello, how can I assist you today regarding your health or any queries you may have?". Always provide point-to-point, concise answers.

    Answer the user query: {user_query} according to the data provided below.

    {content}""")

    # Define the overall chatbot flow
    overall_chat_bot = totle_chat | model | output_parser

    # Process the user query and get AI response
    if user_query:
        Ai_response = overall_chat_bot.invoke({"user_query": user_query, "content": total_data})

        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "ai", "content": Ai_response})

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("You"):
                st.markdown(message["content"])
        else:
            with st.chat_message("AI"):
                st.markdown(message["content"])
    # Chat functionality for SOAP notes


if __name__ == "__main__":
    main()
