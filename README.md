create python environment

python3 -m venv .venv
***


Add python Interpreter in VScode

click view ->command palette-> Add interpreter


Install dependencies:

pip install openai

pip install --upgrade --quiet  langchain langchain-community 
langchainhub langchain-openai

pip install -qU langchain-openai

pip install --upgrade --quiet  docx2txt

pip install streamlit


create new file .env, and add OPENAI_API_KEY & SERPAPI_API_KEY


run below files using streamlit run (file.name) in terminal


sample_llm.py--> only for .txt


sample_llm_with_any_files.py ---> for any files (docx,.txt,.md,excel,pdf)


testing_sample.py ----> works for Internal files (docx,.txt,.md,excel,pdf) and external files (used serpapi)
