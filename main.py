import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # Модель эмбеддингов
from langchain_community.vectorstores import FAISS  # Векторное хранилище FAISS
from langchain.chains import RetrievalQA  # Цепочка для поиска и ответа

book_path = "book.txt"
index_path = "faiss_index"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"


# Проверка наличия корректного FAISS-индекса
def is_faiss_index_valid(path):
    return (
            os.path.isdir(path) and
            os.path.exists(os.path.join(path, "index.faiss")) and
            any(fname.endswith(".pkl") or fname.endswith(".json") for fname in os.listdir(path))
    )


# Предобработка текста и построение индекса (если он отсутствует)
def preprocess_book(book_path=book_path, index_path=index_path):
    # Загрузка текста
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Разделение текста на фрагменты (чанки)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents([text])

    # Добавление индексов чанков в метаданные
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i

    # Создание эмбеддингов и индекс
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(index_path)
    return vectorstore, embedding_model


# Загрузка языковой модели
def load_model(model_name=model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Конфигурация для загрузки модели в 4-битном формате для RTX 4060
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    # Загрузка модели с учетом устройства
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Обертка pipeline Hugging Face
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.3,  # Меньше — детерминированнее
        max_new_tokens=528,
        return_full_text=False,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm


# Запуск цепочки QA на списке вопросов
def run_qa(qa_chain, questions):
    results = []
    for q in questions:
        response = qa_chain.invoke(q)
        results.append({
            "question": q,
            "answer": response["result"],
            "sources": [doc.metadata for doc in response["source_documents"]]
        })
        print(f"Q: {q}\nA: {response['result']}\n")
    return results


if __name__ == "__main__":
    # Предобработка (если индекс отсутствует)
    if not is_faiss_index_valid(index_path):
        print("Создание нового индекса...")
        vectorstore, embedding_model = preprocess_book()
    else:
        print("Индекс найден. Загружаем...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    # Загрузка модели
    llm = load_model()

    # Построение RAG цепочки
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Вопросы для теста
    questions = [
        "What is Kant's definition of pure reason?",
        "How does Kant distinguish between a priori and a posteriori knowledge?",
        "What are the main categories of understanding according to Kant?",
        "What is the role of transcendental aesthetics in Kant's philosophy?",
        "How does Kant argue for the limits of metaphysics?"
    ]

    # Запуск тестов и сохранение результатов
    results = run_qa(qa, questions)
    with open("qa_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
