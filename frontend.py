import streamlit as st
import os
from backend import load_document, chunk_data, create_embedding, ask_answer

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def main():
    st.title("Question with your documents")

    st.subheader('LLM Question-Answering Application 🤖')

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your documents", type=['pdf', 'txt', 'docx'])
        chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:  # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):
                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embedding(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked, and embedded successfully.')

    q = st.text_input('Ask a question about the content of your file:')
    if q:  # if the user entered a question and hit enter
        if 'vs' in st.session_state:  # if there's the vector store (user uploaded, split, and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

if __name__ == "__main__":
    main()
