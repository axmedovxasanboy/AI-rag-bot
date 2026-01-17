import streamlit as st
from core_rag import RAGBot

st.set_page_config(
    page_title="O'zbek AI Yordamchi",
    page_icon="🤖",
    layout="wide"
)

with st.sidebar:
    st.title("O'zbek AI Yordamchi")
    st.markdown("""
    Hujjatlaringiz asosida savollarga javob beruvchi AI yordamchi.

    **Imkoniyatlar:**
    - PDF hujjatlar bilan suhbat
    - Manbalarni ko'rsatish
    - Kontekst asosida javob

    ---
    Developed by Xasanboy  
    RAG-powered assistant
    """)

    if st.button("Suhbatni tozalash"):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
#Hujjatlar bilan muloqot
Savolingizni kiriting va hujjatlaringiz asosida aqlli javob oling.
""")

@st.cache_resource
def load_bot():
    return RAGBot().get_chain()

try:
    rag_chain = load_bot()
except Exception as e:
    st.error("Model yuklanmadi. Quyidagi xatolik yuz berdi:")
    st.code(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Savolingizni yozing...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Javob tayyorlanmoqda..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "Javob topilmadi.")
                context_docs = response.get("context", [])

                st.markdown(answer)

                if context_docs:
                    with st.expander("Manbalar"):
                        unique_sources = set()

                        for doc in response["context"]:
                            source = doc.metadata.get("source", "Nomaʼlum")
                            unique_sources.add(source)

                        for src in unique_sources:
                            st.write(f"- {src}")


            except Exception as e:
                st.error(f"Xatolik yuz berdi: {e}")
                answer = "Xatolik sababli javob berilmadi."

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
