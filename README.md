**Студент: Нестеренко Даниил
Группа: 932302

### Цель работы

Освоить базовые принципы Retrieval-Augmented Generation (RAG) и применить их для построения простой системы, способной отвечать на вопросы по содержанию конкретной книги. Работа направлена на создание ресурсоэффективного прототипа, который позволяет извлекать релевантные фрагменты текста и формировать осмысленные ответы с помощью языковой модели.

---

### Ход работы

#### 1. Выбор и обоснование базы данных для хранения эмбеддингов

В качестве хранилища векторных представлений (эмбеддингов) был выбран FAISS (Facebook AI Similarity Search). Основные причины выбора:

- Высокая производительность при поиске по большим объемам векторов.
    
- Поддержка локального использования без необходимости развертывания серверов.
    
- Совместимость с LangChain и HuggingFace.
    
- Простота сериализации и десериализации индексов.

#### 2. Выбор и обоснование языковой модели

Для генерации ответов была выбрана модель `mistralai/Mistral-7B-Instruct-v0.2`, размещённая на Hugging Face. Обоснование выбора:

- Поддержка инструкционного режима (Instruct), что позволяет формулировать более осмысленные ответы на вопросы.
    
- Возможность запуска локально с использованием библиотеки `transformers`.
    
- Поддержка квантования (BitsAndBytesConfig), что существенно снижает требования к оперативной памяти при использовании GPU.
    

#### 3. Реализация загрузки и предварительной обработки текстового документа

- Исходный текст книги загружается из файла `book.txt`.
    
- Для разбиения текста на части используется `RecursiveCharacterTextSplitter` с параметрами `chunk_size=500` и `chunk_overlap=100`.
    
- Каждому фрагменту присваивается индекс в метаданных.
    
- С помощью модели `sentence-transformers/all-MiniLM-L6-v2` создаются эмбеддинги и сохраняются в индекс FAISS.
    
- Проверяется наличие ранее созданного индекса, чтобы избежать повторной генерации.
    

#### 4. Построение и запуск RAG-пайплайна

- Загружается локальная языковая модель с использованием пайплайна HuggingFace (`text-generation`).
    
- Создаётся цепочка `RetrievalQA` из LangChain с использованием метода `from_chain_type`, где:
    
    - `llm` — языковая модель,
        
    - `retriever` — объект поиска по эмбеддингам с top-k=3,
        
    - `return_source_documents=True` — возвращение исходных документов.

#### 5. Тестирование пайплайна на собственных примерах вопросов

Были заданы следующие вопросы по философии Канта:

1. What is Kant's definition of pure reason?
2. How does Kant distinguish between a priori and a posteriori knowledge?
3. What are the main categories of understanding according to Kant?
4. What is the role of transcendental aesthetics in Kant's philosophy?
5. How does Kant argue for the limits of metaphysics?

Система корректно обрабатывала каждый запрос, находила релевантные фрагменты текста и формировала ответы, подтверждённые источниками.

Результаты сохраняются в JSON-файл `qa_results.json` для последующего анализа.

---

### Блок-схема программы

![[Pasted image 20250531020452.png]]

---

### Код программы

```
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
```

---

### Ответы программы и их анализ

#### ❓Q: _What is Kant's definition of pure reason?_

**Ответ:**

> Pure reason refers to the use of reason to gain knowledge about concepts and the world, independent of sensory experience... its ultimate end is the _summum bonum_... it has a canon and architectonic...

**✅ Верно частично:**

- **Правильно:** Упоминание, что "pure reason" стремится к знанию **вне опыта**, верно. Также Канту важно, что чистый разум сам формирует идеи, не опираясь на чувства.
    
- **Неточности:**
    
    - **"to gain knowledge about the world"** — не совсем точно. Кант утверждает, что чистый разум **не может** дать знание о мире сам по себе, без опыта.
        
    - **summum bonum** — это **не конечная цель разума в "Критике чистого разума"**, а скорее в "Критике практического разума" (этика). Здесь путаница между двумя "Критиками".
        
    - Упоминание **"canon"** и **"architectonic"** верно, но это вспомогательные термины, а не суть определения.
        

**📘 Корректный вариант:**

> For Kant, pure reason is the faculty that seeks knowledge independently of experience. It deals with a priori principles and ideas such as the soul, the world as a whole, and God — but ultimately cannot provide certain knowledge about them. The critical task is to explore the limits of this faculty.

---

#### ❓Q: _How does Kant distinguish between a priori and a posteriori knowledge?_

**Ответ:**

> A priori knowledge comes from general rules borrowed from experience...

**❌ Ошибочно:**

- **Кант прямо противопоставляет** _a priori_ и _a posteriori_ по **источнику**:
    
    - _A priori_ — независимое от опыта знание.
        
    - _A posteriori_ — основано на опыте.
        
- **"borrowed from experience"** — **противоречит Канту**. Это скорее определение _a posteriori_.
    
- Пример "каждое изменение имеет причину" — **верен**, но его происхождение объясняется у Канта как _синтетическое априори_, а не просто как правило из опыта.
    

**📘 Корректный вариант:**

> Kant distinguishes a priori knowledge as independent of experience, arising from pure reason, while a posteriori knowledge is derived from empirical observation. A priori truths are necessary and universal (e.g., "7 + 5 = 12"), while a posteriori truths depend on individual cases.

---

#### ❓Q: _What are the main categories of understanding according to Kant?_

**Ответ:**

> The categories include substance, causality, existence, necessity...

**✅ Частично верно, но поверхностно и неаккуратно:**

- Кант выделяет **12 категорий**, организованных по **четырём группам**:
    
    1. Quantity: Unity, Plurality, Totality
        
    2. Quality: Reality, Negation, Limitation
        
    3. Relation: Inherence (Substance/Accident), Causality, Reciprocity
        
    4. Modality: Possibility, Existence, Necessity
        
- Упомянуто **несколько категорий**, но **не указана структура** и **термины даны не в точной форме**.
    

**📘 Корректный вариант:**

> Kant proposed 12 pure concepts of the understanding (categories), grouped under Quantity, Quality, Relation, and Modality. These categories are necessary for synthesizing experience and making sense of phenomena.

---

#### ❓Q: _What is the role of transcendental aesthetics in Kant's philosophy?_

**Ответ:**

> Not explicitly mentioned... But it deals with the nature and limits of sensory experience and forms of intuition...

**✅ Верно по сути:**

- Правильно говорится, что **трансцендентальная эстетика** занимается **восприимчивостью (sensibility)** и **формами чувственного восприятия** — пространством и временем.
    
- Однако:
    
    - Утверждение "не упоминается в тексте" (видимо, от модели) не информативно.
        
    - Концовка про "aesthetic experience" сбивает с толку — **это не про красоту**, а про **формы чувственного восприятия**.
        

**📘 Корректный вариант:**

> Transcendental aesthetics is the part of Kant's philosophy that investigates the a priori forms of sensibility — space and time — which structure all our experiences. It explains how objects are given to us, as opposed to how they are thought.

---

#### ❓Q: _How does Kant argue for the limits of metaphysics?_

**Ответ:**

> Metaphysics cannot form the foundation of religion... but it is a bulwark... its limits are necessary for perfection...

**✅ Частично верно, но путано:**

- Кант действительно ограничивает метафизику: **она не может дать знания о "вещах в себе"** — только о явлениях.
    
- Верно сказано, что **разум склонен к иллюзиям**, если выходит за границы опыта (диалектика).
    
- Но:
    
    - **"bulwark of religion"** — не главное утверждение Канта.
        
    - Заключение про "размер и число" — не имеет прямого отношения к критике метафизики.
        

**📘 Корректный вариант:**

> Kant argues that traditional metaphysics oversteps the bounds of reason by trying to know things beyond possible experience (e.g., God, the soul). His critical philosophy sets limits to pure reason, showing that such knowledge is impossible, though metaphysical ideas can have a regulative role.

---

####  Общая оценка:

|Вопрос|Точность|Комментарий|
|---|---|---|
|1|7/10|Верная суть, но терминология смешана с практической философией.|
|2|4/10|Сильно искажена трактовка a priori и a posteriori.|
|3|6/10|Перечислены категории, но неполно и без структуры.|
|4|7/10|Почти верно, но с путаницей в терминах.|
|5|6/10|Общая идея верна, но без чёткого фокуса на критику "догматической" метафизики.|

---

### Выводы

В результате лабораторной работы:

- Освоены принципы Retrieval-Augmented Generation (RAG).
- Реализована система, способная обрабатывать текст книги и отвечать на вопросы с использованием векторного поиска и генерации текста.
- Продемонстрированы возможности LangChain, Hugging Face и FAISS для построения гибкого и расширяемого RAG-прототипа.
- Получен практически применимый опыт создания интеллектуального помощника на базе локальных языковых моделей и эмбеддингов.

---

### Приложение

- `book.txt` — исходный текст
    
- `faiss_index/` — папка с сохранённым индексом
    
- `qa_results.json` — результаты тестирования вопросов
    
- `main.py` — основной код системы
