import os
import streamlit as st

from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate, FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()
# OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0)
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

user_input = st.text_area("Enter your text here")

prompt_template = """Write a concise bullet point summary of the following:
{text}

CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["text"])

chain = load_summarize_chain(llm,
                           chain_type="stuff",
                           prompt=BULLET_POINT_PROMPT)

if st.button('Generate') and user_input:
    docs = text_splitter.create_documents([user_input])
    output_summary = chain.run(docs)
    st.write('Summary Article')
    st.text(output_summary)