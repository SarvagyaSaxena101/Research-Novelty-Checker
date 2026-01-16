<div align="center">

# 🔬 Research Novelty Checker 🔬

# Deployed Link - 
### https://researchnoveltyai.streamlit.app/
May take sometime to load buddy but will surely work


The **Research Novelty Checker** is your AI-powered lab partner, designed to give your research paper a preliminary review before you submit it to the unforgiving world of peer review. It uses a multi-faceted approach to analyze your work, checking for novelty against existing literature, flagging statistical inconsistencies, and even generating a full-fledged review.

---

## ✨ Key Features

-   **📄 PDF Analysis:** Simply upload your research paper in PDF format.
-   **💡 Novelty Scoring:** Compares your paper's abstract against the vast arXiv database to determine its uniqueness. See how your work stacks up against the top 5 most similar papers.
-   **📊 Statistical Sanity Check:** Employs a Large Language Model (LLM) to read through your paper and flag any claims, figures, or results that seem inconsistent or poorly supported.
-   **✍️ Methodology Clarity Assessment:** Get an AI-generated score (1-10) on how clear and reproducible your methodology is.
-   **💯 Confidence Score:** A single, intuitive metric that combines all analyses to give you a "Review Confidence Score," predicting how well your paper might be received.
-   **🤖 Automated Peer Review:** Receive a comprehensive, structured review of your paper, complete with a summary, strengths, weaknesses, and suggestions for improvement, just like a real conference reviewer would provide.

---

## ⚙️ How It Works

This tool is built on a modern stack of AI and data science technologies:

-   **Frontend:** [Streamlit](https://streamlit.net/) provides the interactive and easy-to-use web interface.
-   **Core Logic:** [LangChain](https://www.langchain.com/) orchestrates the interactions between the language models and the different components of the application.
-   **AI Brain:** The analysis is powered by LLMs accessed through the [OpenRouter](https://openrouter.ai/) API, allowing for flexibility and access to state-of-the-art models.
-   **Novelty Search:**
    -   [Sentence-Transformers](https://www.sbert.net/) generates vector embeddings for the paper abstracts.
    -   [FAISS](https://faiss.ai/) (Facebook AI Similarity Search) performs a blazing-fast similarity search on these embeddings.
    -   The [arXiv API](https://info.arxiv.org/help/api/index.html) is used to fetch abstracts from the latest research papers.

---

## 🚀 Setup and Usage

Get your own instance of the Research Novelty Checker running in a few simple steps.

### 1. Clone the Repository
```bash
cd Research-Novelty-Checker
```

### 2. Create and Activate a Virtual Environment
This project uses a Python virtual environment to manage dependencies.

**On Windows:**
```bash
python -m venv novelty
.\novelty\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv novelty
source novelty/bin/activate
```

### 3. Install Dependencies
Install all the required packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
The application requires an API key from [OpenRouter](https://openrouter.ai/) to function.

-   Open the `app.py` file.
-   You will be prompted to enter your API key in the web interface sidebar when you run the app.

### 5. Run the Streamlit App
Launch the application using Streamlit.
```bash
streamlit run app.py
```
Your web browser should open with the application running!

---

## ⚠️ Disclaimer

This is an AI-powered tool and is not a substitute for a thorough human review. The analyses and reviews generated are based on the patterns the AI has learned from vast amounts of data and should be used as a supplementary tool to help you refine your manuscript. Always use your own judgment and expertise.