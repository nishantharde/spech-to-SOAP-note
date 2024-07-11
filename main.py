import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate  # Ensure this import is correct based on your library

# Load environment variables
load_dotenv()

# Read the contents of the file
with open('data.txt', 'r') as file:
    content = file.read()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.header('Ask Anything about SOAP notes')
st.text('VKAPS It Solutions Pvt Ltd.')
model = ChatOpenAI(model="gpt-3.5-turbo")

output_parser = StrOutputParser()

# Get user query input
user_query = st.chat_input("Ask your query here about the given patient details...")

# Create the chat prompt template
totle_chat = ChatPromptTemplate.from_template(""" 
answer the user query {user_query} according to the data provided below. Remember that the whole data provided below is of the user who is asking the query to you. 
"You are a helpful AI assistant who is very humble and able to respond to every query of the user related to the user's health."
{content}
""")

# Define the overall chatbot flow
overall_chat_bot = totle_chat | model | output_parser

# Process the user query and get AI response
if user_query:
    Ai_response = overall_chat_bot.invoke({"user_query": user_query, "content": content})
    with st.chat_message("You"):
        st.markdown(user_query)
    # Display AI's answer
    with st.chat_message("AI"):
        st.markdown(Ai_response)
    
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

