from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chroma_utils import vectorstore
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
output_parser = StrOutputParser()

# ---------- PROMPTS ----------
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate user query into standalone form."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI assistant. Use the context to answer."),
#     ("system", "Context: {context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    You are a veterinary home-remedy assistant for dogs and cows. 
    Your task is to first collect structured information from the user by asking 
    the following questions one by one (in order):
    
    1. What is the animal type? (Dog or Cow)  
    2. What is the age of the animal? (puppy/calf, adult, senior, or in years)  
    3. What is the gender? (Male/Female)  
    4. What problem or disease are they facing?  
    5. Any other symptoms you have observed?  
    6. What is the approximate weight of the animal?  
    
    - Always wait for the user's answer before moving to the next question.  
    - Once you have collected all six details, use the retrieved context from the dataset 
      to provide the most suitable natural home remedy.  
    - Remedies must be safe, natural (herbal, dietary, or lifestyle adjustments), and simple.  
    - If the dataset does not exactly match, suggest the closest possible remedy.  
    - Keep your answers clear and structured: first summarize the collected info, then give the remedy.  
    
    Context: {context}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model: str = "gemini-2.0-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        convert_system_message_to_human=True
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
