# Instructions on how to use the chatbot

## File structure
km_chatbot/
├── app.py             # <-- Main file
├── requirements.txt
├── chroma_store/      # <-- Your vector DB folder
└── README.md          # <-- Instructions


1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Run `ollama serve`
4. Run `python files_to_json.py`
3. Open a new terminal and run `streamlit run app.py`


