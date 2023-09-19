import streamlit as st
import openai
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI , HuggingFaceHub
from dotenv import load_dotenv

# Set the page layout to a wider layout
st.set_page_config(layout="wide")


# analyse resume function will extract deatails like experience, skills, designation by custom prompting resume text through huggingface llm.

def analyze_resume(job_desc, resume, options):
    df = analyze_str(resume, options)
    df_string = df.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x).to_string(index=False)
    st.write("Analyzing with Huggingface..")
    summary_question = f"Job requirements: {{{job_desc}}}" + f"Resume summary: {{{df_string}}}" + "Please return a summary of the candidate's suitability for this position (limited to 200 words);'"
    summary = ask_openAI(summary_question)
    df.loc[len(df)] = ['Summary', summary]
    score_question = f"Job requirements: {{{job_desc}}}" + f"Resume summary: {{{df.to_string(index=False)}}}" + "Please return a matching score (0-100) for the candidate for this job, please score accurately to facilitate comparison with other candidates, '"
    score = ask_openAI(score_question)
    df.loc[len(df)] = ['Match Score', score]

    return df

# we are using openai davinchi text model to respond  concisely and more accurately 

def ask_openAI(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].text.strip()

# splitting data into chunks will help to overcome token limit condition for llm model.

def analyze_str(resume, options):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(resume)

    # loading api key from virtual environment

    load_dotenv()


    # we are using instructor embeddings to vectorise text data and save locally using faiss.
    # unlike open ai embeddings , instructor embeddings will vectorise data on local machine.

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    

    df_data = [{'option': option, 'value': []} for option in options]
    st.write("Fetching information")

    # Create a progress bar and an empty element
    progress_bar = st.progress(0)
    option_status = st.empty()

    for i, option in tqdm(enumerate(options), desc="Fetching information", unit="option", ncols=100):
        question = f"What is this candidate's {option}? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'"
        
        # similarity search will calcaulate similarity score between cv and job description
      
        docs = knowledge_base.similarity_search(question)
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        # llm = OpenAI(temperature=0.3, model_name="text-davinci-003", max_tokens="2000")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question )
        df_data[i]['value'] = response
        option_status.text(f"Looking for information: {option}")

        # Update the progress bar
        progress = (i + 1) / len(options)
        progress_bar.progress(progress)

    df = pd.DataFrame(df_data)
    st.success("Resume elements retrieved")
    return df

# Set the page title
st.title("Resume Matching with Job Descriptions")
st.subheader("Langchain + Huggingface")

# Set default job description and resume information
default_jd = "Job description"
default_resume = "Resume"

# Enter job description
jd_text = st.text_area("【Job Description】", height=100, value=default_jd)

# Enter resume information
resume_text = st.text_area("【Candidate Resume】", height=100, value=default_resume)

# Parameter input
options = ["Name", "Contact Number", "Gender", "Age", "Years of Work Experience (Number)", "Highest Education", "Undergraduate School Name", "Master's School Name", "Employment Status", "Current Position", "List of Past Employers", "Technical Skills", "Experience Level", "Management Skills"]
selected_options = st.multiselect("Please select options", options, default=options)

# Analyze button
if st.button("Start Analysis"):
    df = analyze_resume(jd_text, resume_text, selected_options)
    st.subheader("Overall Match Score: "+ df.loc[df['option'] == 'Match Score', 'value'].values[0])
    st.subheader("Detailed Display:")
    st.table(df)
