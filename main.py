import streamlit as st
import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain import PromptTemplate

MODEL_NAME = "TheBloke/Llama-2-7b-Chat-GPTQ"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 256
    generation_config.temperature = 0.5
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

llm = load_model_and_tokenizer()

# Load and process your data outside the main function
loader = UnstructuredMarkdownLoader("/content/questions.csv")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

# Consider if this part can be preprocessed or cached
db = Chroma.from_documents(texts, embeddings, persist_directory="db")

prompt_template = """
<s>[INST] <<SYS>>
You are expert in Human development index. You have basic to advance knowledge to answer related to human development index (HDI).
<</SYS>>

{context}

{question} [/INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

def main():
    st.title("HDI")
    q = st.text_input("Question")
    if st.button("Submit"):
        result = qa_chain(q)
        st.write(result["result"].strip())

if __name__ == "__main__":
    main()