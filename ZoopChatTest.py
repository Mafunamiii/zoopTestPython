from langchain_text_splitters import RecursiveCharacterTextSplitter

from Custom_Logger import setup_logger
import os
from dotenv import load_dotenv
from langchain_core.prompts import  PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from enum import Enum
from langchain_community.document_loaders import WebBaseLoader
from nemoguardrails import LLMRails, RailsConfig

logger = setup_logger()
logger.info("logger setup.")
load_dotenv()
logger.info("Loaded environment variables from .env file.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger.info("Read OPENAI_API_KEY environment variable.")
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY":
    logger.error("Please set your OPENAI_API_KEY environment variable.")
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")


# Set the environment variable for further use in your application
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
logger.info("Set OPENAI_API_KEY environment variable.")

# Your other code here
class AiModel(Enum):
    GPT_4_O_MINI = "gpt-4o-mini"
logger.info(f"Model: {AiModel.GPT_4_O_MINI.value}")
model = ChatOpenAI(
    model=AiModel.GPT_4_O_MINI.value,
    temperature=0.3,
    max_tokens=300,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5
)

def displayMark(content):
    logger.info(f"Content: {content}")
    if isinstance(content, dict) and 'content' in content:
        logger.info("Content is a valid response.")
        response_content = content['content'][:4000]  # Truncate response if too long
        st.markdown(response_content)
        logger.info(f"Response Content: {response_content}")
        model_name = content.get('response_metadata', {}).get('model_name', 'Unknown')
        st.markdown(f"**Model:** {model_name}")
    else:
        logger.error("Invalid response format.")
        st.error("Invalid response format.")

config = RailsConfig.from_path("./config")
rails = LLMRails(config, llm=model)


def chunk_text(url: str, chunk_size: int, chunk_overlap):
    logger.info(f"URL: {url}")
    logger.info(f"Chunk Size: {chunk_size}")
    logger.info(f"Chunk Overlap: {chunk_overlap}")

    loader = WebBaseLoader(url)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split {len(chunks)} chunks.")
    cleaned_chunks = []
    for chunk in chunks:
        cleaned_content = chunk.page_content.replace("\n", " ")  # Replace newlines with space
        cleaned_content = ' '.join(cleaned_content.split())  # Remove extra spaces
        cleaned_chunks.append(cleaned_content)
    logger.info(f"Cleaned {len(cleaned_chunks)} chunks.")
    return cleaned_chunks


def chat_with_ai(query: str, url: str):
    logger.info(f"Prompt: {query}")

    # Load and chunk the context from the URL
    context_chunks = chunk_text(url=url, chunk_size=2000, chunk_overlap=1000)
    logger.info(f"Context chunks: {context_chunks}")

    # Limit the number of context chunks to fit within token limits
    max_context_length = 5000  # Adjust this based on your model's limit
    context = ' '.join(context_chunks)[:max_context_length]
    logger.info(f"Final Context (truncated): {context}")

    # Create the prompt with a clear instruction
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="Based on this context: {context} \n\n Question: {query}\n Answer:"
    )

    # Format the prompt
    prompt = prompt_template.format(context=context, query=query)
    logger.info(f"Prompt: {prompt}")

    # Check if the prompt exceeds the max length
    max_prompt_length = 16000
    if len(prompt) > max_prompt_length:
        logger.error("Prompt exceeds the maximum length.")
        st.error("The generated prompt is too long. Try reducing the context or query length.")
        return {"content": "Prompt exceeds maximum length. Try reducing the input size."}

    # Modify the prompt for conciseness
    modified_prompt = f"Make the message short and concise. {prompt}"
    logger.info(f"Modified Prompt: {modified_prompt}")

    # Create a list of messages for the LLMRails
    messages = [
        {"role": "user", "content": modified_prompt}
    ]

    # Generate the response
    response = rails.generate(messages=messages)
    logger.info(f"Response: {response}")

    return response



st.title("Zoop AI!")

user_input = st.text_input("Enter your question:")
url = st.text_input("Enter the URL for context (optional):", "https://en.wikipedia.org/wiki/Python_(programming_language)")

if st.button("Send"):
    logger.info("Send button clicked.")
    if user_input:
        logger.info(f"User Input: {user_input}")
        response = chat_with_ai(user_input, url)
        st.write("AI Response:")
        logger.info(f"AI Response: {response}")
        displayMark(response)
    else:
        st.write("Please enter a question.")