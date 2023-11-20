###############################################################
# +=========================================================+ #
# |     AI Powered by Openai Api - PDF Asking About         | #
# +=========================================================+ #
# | Author   : JOSE TEOTONIO DA SILVA NETO [TEO]            | #
# | Objective: Build a simple using openai Api              | #
# | Version  : 1.0.0.0                                      | #
# +=========================================================+ #
# | Name   | Changed At | Description                       | #
# +=========================================================+ #
# | Teo    | 23/09/2023 | Build Starter Version             | #
# +=========================================================+ #
###############################################################

# +=========================================================+ #
# | Libraries necessaries to execute current project        | #
# +=========================================================+ #
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# +=========================================================+ #
# | Function to importing, idex content and ask PDF file    | #
# +=========================================================+ #
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # +=========================================================+ #
    # | Upload File PDF                                         | #
    # +=========================================================+ #
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # +=========================================================+ #
    # | Extrat all content from Uploaded File PDF               | #
    # +=========================================================+ #
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # +=========================================================+ #
        # | Split all content into chunks Uploaded File PDF         | #
        # +=========================================================+ #
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # +=========================================================+ #
        # | Create embeddings from Uploaded File PDF                | #
        # +=========================================================+ #
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # +=========================================================+ #
        # | Show User imput to ask about current file imported      | #
        # +=========================================================+ #
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
