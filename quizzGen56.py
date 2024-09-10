import tempfile
import streamlit as st
import fitz  # PyMuPDF
from google.cloud import aiplatform
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI, RateLimitError
import time

# Configuration for embedding client
embed_config = {
    "model_name": "ModelName",
    "project": "ProjectName",
    "location": "us-central1"
}

# API key
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Initialize session state variables
if 'question_bank' not in st.session_state:
    st.session_state['question_bank'] = []

if 'display_quiz' not in st.session_state:
    st.session_state['display_quiz'] = False

if 'question_index' not in st.session_state:
    st.session_state['question_index'] = 0

if 'quiz_manager' not in st.session_state:
    st.session_state['quiz_manager'] = None

# Display the quiz if the display_quiz flag is set
if st.session_state['display_quiz']:
    st.header("Quiz Time!")
    # Retrieve the current question using the quiz manager
    quiz_manager = st.session_state['quiz_manager']
    index_question = quiz_manager.get_question_at_index(st.session_state['question_index'])
    st.subheader("Generated Quiz Question:")
    st.write(index_question)
    
    if quiz_manager.question_type == "multiple choice":
        # Unpack the choices from the question onto a radio button
        question_text, *choices = index_question.split("\n")
        selected_choice = st.radio(question_text, choices, key=index_question, index=None)
    else:
        st.text_area("Your Response", key=f"response_{st.session_state['question_index']}")

    # Step 8: Add a button to navigate to the next question
    if st.button("Next Question"):
        quiz_manager.next_question_index(direction=1)
        index_question = quiz_manager.get_question_at_index(st.session_state['current_question_index'])
        st.subheader("Generated Quiz Question:")
        st.write(index_question)
        
        if quiz_manager.question_type == "multiple choice":
            # Unpack the choices from the question onto a radio button
            question_text, *choices = index_question.split("\n")
            selected_choice = st.radio(question_text, choices, key=index_question, index=None)
        else:
            st.text_area("Your Response", key=f"response_{st.session_state['question_index']}")

# Define the remaining classes and functions

# Mock implementation of EmbeddingClient class
class EmbeddingClient:
    def __init__(self, model_name, project, location):
        self.model_name = model_name
        self.project = project
        self.location = location

    def get_embedding(self, text):
        return [ord(char) for char in text]  # Simple mock: convert each character to its ASCII value

# Mock implementation of Chroma class
class Chroma:
    @staticmethod
    def from_documents(documents):
        return {"documents": documents}
    
    @staticmethod
    def search(query, collection):
        return [doc for doc in collection["documents"] if query.lower() in doc["text"].lower()]

class PDFProcessor:
    def __init__(self, embed_config):
        self.model_name = embed_config["model_name"]
        self.project = embed_config["project"]
        self.location = embed_config["location"]
        self.pages = []
        self.client = EmbeddingClient(model_name=self.model_name, project=self.project, location=self.location)

    def get_text_embedding(self, text):
        return self.client.get_embedding(text)

    def ingest_documents(self, uploaded_files):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            with st.spinner(f'Processing {uploaded_file.name}'):
                document = fitz.open(temp_file_path)
                extracted_pages = [page.get_text("text") for page in document]
                self.pages.extend(extracted_pages)
                for i, page_content in enumerate(extracted_pages):
                    st.write(f"Page {i + 1}:")
                    st.write(page_content)
                document.close()

class ChromaCollectionCreator:
    def __init__(self, document_processor, temperature=0.7, max_output_tokens=500):
        self.document_processor = document_processor
        self.collection = None
        self.model_name = "gemini-pro"
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.llm_initialized = False

    def create_chroma_collection(self):
        if not self.document_processor.pages:
            st.error("No documents have been processed.")
            return
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        for page in self.document_processor.pages:
            chunks = text_splitter.split_text(page)
            all_chunks.extend(chunks)
        documents = [{"text": chunk, "embedding": self.document_processor.get_text_embedding(chunk)} for chunk in all_chunks]
        self.collection = Chroma.from_documents(documents)
        st.success("Chroma collection created successfully!")
        st.write(self.collection)
        st.session_state['chroma_collection'] = self.collection

    def search_chroma_collection(self, query):
        if 'chroma_collection' not in st.session_state:
            st.error("No Chroma collection found in session state.")
            return []
        self.collection = st.session_state['chroma_collection']
        return Chroma.search(query, self.collection)

    def initialize_llm(self):
        self.llm = aiplatform.gapic.ModelServiceClient()
        self.llm_initialized = True

class QuizGenerator:
    def __init__(self, topic, num_questions, question_type, chroma_collection):
        self.topic = topic
        self.num_questions = num_questions
        self.question_type = question_type
        self.chroma_collection = chroma_collection

    def generate_question_with_vectorstore(self, context):
        if not self.chroma_collection:
            st.error("Vectorstore is not initialized.")
            return []
        search_results = Chroma.search(context, self.chroma_collection)
        if not search_results:
            st.error(f"No relevant documents found for the context: {context}")
            return []
        context_text = "\n".join([doc['text'] for doc in search_results])
        prompt = f"Generate a unique {self.question_type} quiz question based on the following context:\n\n{context_text}"
        st.write(f"Generating a question on the context: {context} using GPT-3")
        st.write(f"Prompt: {prompt}")

        attempts = 0
        max_attempts = 5  # Define max attempts to retry in case of rate limit error

        while attempts < max_attempts:
            try:
                # Use OpenAI GPT-3 to generate questions
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,  # Adjust as needed
                    temperature=0.7
                )

                generated_question = response.choices[0].message.content.strip()
                return generated_question  # Return the generated question

            except RateLimitError as e:
                st.warning(f"Rate limit error: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                attempts += 1

        st.error("Failed to generate questions due to rate limits.")
        return ""

    def validate_question(self, question, question_bank):
        return question not in question_bank

    def generate_quiz(self):
        combined_unique_questions = []
        st.write("Initialized empty list for combined unique quiz questions.")
        
        # Extract unique meaningful contexts
        contexts = [chunk['text'] for chunk in self.chroma_collection["documents"][:self.num_questions]]
        attempts = 0  # To prevent infinite loops
        max_attempts = self.num_questions * 2  # Allow twice the number of attempts as the number of questions

        for context in contexts:
            if len(combined_unique_questions) >= self.num_questions:
                break
            while attempts < max_attempts:
                question = self.generate_question_with_vectorstore(context)
                if question and self.validate_question(question, combined_unique_questions):
                    # Remove "Quiz Question:" and "Unique quiz question:" from the question
                    question = question.replace("Quiz Question:", "").replace("Unique quiz question:", "").strip()
                    combined_unique_questions.append(question)
                    break  # Stop if we have a unique question for this context
                attempts += 1

        # Store the questions in the instance variable 'questions'
        self.questions = combined_unique_questions

        if len(combined_unique_questions) < self.num_questions:
            st.error("Could not generate the desired number of unique questions. Try adjusting the topic or input parameters.")
        else:
            st.write("Final List of Unique Quiz Questions:")
            for i, question in enumerate(combined_unique_questions, 1):
                st.write(f"Question {i}")  # Print the question number
                st.write(question)  # Print the actual question
        return combined_unique_questions

class QuizManager:
    def __init__(self, questions, question_type):
        self.questions = questions  # Store the provided list of quiz question objects
        self.total_questions = len(questions)  # Calculate and store the total number of questions
        self.question_type = question_type

    def next_question_index(self, direction):
        # Retrieve the current question index from Streamlit's session state
        if 'current_question_index' not in st.session_state:
            st.session_state['current_question_index'] = 0
        
        # Adjust the index based on the provided direction
        new_index = (st.session_state['current_question_index'] + direction) % self.total_questions

        # Update the session state with the new index
        st.session_state['current_question_index'] = new_index
        return new_index
    
    def get_question_at_index(self, index):
        return self.questions[index]

# Main execution flow for processing files and handling inputs
pdf_processor = PDFProcessor(embed_config=embed_config)
uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_processor.ingest_documents(uploaded_files)

if pdf_processor.pages:
    st.write("Extracted Pages Content:")
    for i, page_content in enumerate(pdf_processor.pages):
        st.write(f"Page {i + 1}:")
        st.write(page_content)

if st.button("Generate Embedding for 'Hello World'"):
    embedding = pdf_processor.get_text_embedding("Hello World")
    st.write("Embedding for 'Hello World':")
    st.write(embedding)

# Instantiate ChromaCollectionCreator using the initialized PDFProcessor
chroma_creator = ChromaCollectionCreator(pdf_processor, temperature=0.7, max_output_tokens=500)

# Sidebar for inputs
st.sidebar.header("Quiz Configuration")
with st.sidebar.form(key='quiz_form'):
    quiz_topic = st.text_input("Enter the quiz topic")
    num_questions = st.slider("Select the number of questions", min_value=1, max_value=50, value=10)
    question_type = st.radio("Select the question type", ("multiple choice", "open ended"))
    submit_button = st.form_submit_button(label='Generate Quiz')

if submit_button:
    st.write(f"Quiz Topic: {quiz_topic}")
    st.write(f"Number of Questions: {num_questions}")
    chroma_creator.create_chroma_collection()
    chroma_creator.initialize_llm()
    
    # Step 3: Initialize a QuizGenerator class
    quiz_generator = QuizGenerator(quiz_topic, num_questions, question_type, st.session_state['chroma_collection'])
    unique_questions = quiz_generator.generate_quiz()
    
    st.write("Final List of Unique Quiz Questions:")
    for i, question in enumerate(unique_questions, 1):
        st.write(f"Question {i}")  # Print the question number
        st.write(question)  # Print the actual question

    # Step 5: Set display_quiz flag to True
    st.session_state['display_quiz'] = True

    # Step 6: Set the question_index to 0
    st.session_state['question_index'] = 0

    # Step 7: Create an instance of QuizManager with the generated questions and store it in session state
    st.session_state['quiz_manager'] = QuizManager(unique_questions, question_type)

# Sidebar for Chroma collection search
st.sidebar.header("Search Chroma Collection")
query = st.sidebar.text_input("Enter a query related to the quiz topic")

if st.sidebar.button("Search"):
    if query:
        results = chroma_creator.search_chroma_collection(query)
        if results:
            st.write("Search Results:")
            for result in results:
                st.write(result["text"])
        else:
            st.write("No results found.")
