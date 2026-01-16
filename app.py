import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from pypdf import PdfReader
import re
import arxiv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.title("🔬 Research Novelty Checker")

@st.cache_resource
def get_llm(api_key):
    return ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key
    )

@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_title_and_abstract(text):
    lines = text.split('\n')
    title = lines[0]
    abstract_match = re.search(r'Abstract\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip().split('\n\n')[0]
    else:
        abstract = " ".join(lines[1:10])
    return title, abstract

def check_novelty(title, abstract, model):
    st.subheader("Novelty Analysis")
    try:
        search = arxiv.Search(query=f'ti:"{title}"', max_results=10, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
        if not results:
            st.warning("Could not find any similar papers on arXiv based on the title.")
            return 1.0, 0
        
        st.info(f"Found {len(results)} potentially similar papers. Comparing abstracts...")
        arxiv_abstracts = [result.summary for result in results]
        uploaded_abstract_embedding = model.encode([abstract])
        arxiv_abstract_embeddings = model.encode(arxiv_abstracts)
        
        d = arxiv_abstract_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(arxiv_abstract_embeddings)
        
        k = min(5, len(results))
        distances, indices = index.search(uploaded_abstract_embedding, k)
        
        st.markdown("---")
        st.subheader("Top 5 Most Similar Papers on arXiv:")
        
        # Higher similarity should lead to lower novelty_score.
        # Let's define novelty as 1 - similarity.
        top_similarity = 1 / (1 + distances[0][0])
        novelty_score = 1 - top_similarity


        for i in range(k):
            sim_index = indices[0][i]
            sim_score = 1 / (1 + distances[0][i])
            st.markdown(f"**{sim_score:.2f} Similarity** - [{results[sim_index].title}]({results[sim_index].entry_id})")
            st.text(f"Published: {results[sim_index].published.date()}")
            with st.expander("Show Abstract"):
                st.write(results[sim_index].summary)
        return novelty_score, len(results)

    except Exception as e:
        st.error(f"An error occurred during novelty analysis: {e}")
        return 0.5, 0

def detect_statistical_inconsistencies(text, llm):
    st.subheader("Statistical Inconsistency Analysis")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    prompt_template = """
    You are an expert at analyzing scientific papers. Your task is to identify any statistical claims, figures, or results in the following text that seem inconsistent, contradictory, or inadequately supported.

    Here is a chunk of the research paper:
    ---
    {text_chunk}
    ---

    Please list any potential statistical inconsistencies you find. For each, describe the issue and why it seems inconsistent. If you find no inconsistencies, please state that clearly.
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text_chunk"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    with st.spinner("Analyzing text for statistical inconsistencies..."):
        all_inconsistencies = []
        for chunk in chunks:
            response = chain.run(chunk)
            if "no inconsistencies" not in response.lower():
                all_inconsistencies.append(response)

        if not all_inconsistencies:
            st.success("No apparent statistical inconsistencies were found.")
        else:
            st.warning("Potential statistical inconsistencies found:")
            for inconsistency in all_inconsistencies:
                st.write(inconsistency)
    return len(all_inconsistencies)

def get_methodology_clarity(text, llm):
    st.subheader("Methodology Clarity Analysis")
    prompt_template = """
    Based on the 'Methods' or 'Methodology' section of the following paper, please rate the clarity and reproducibility of the methodology on a scale from 1 to 10. 
    1 means extremely unclear and not reproducible. 10 means perfectly clear and easily reproducible.
    Please provide a score and a brief justification. Return only the number and the justification on a new line.

    Paper Text:
    ---
    {paper_text}
    ---
    Clarity Score (1-10):
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["paper_text"])
    chain = LLMChain(llm=llm, prompt=prompt)

    with st.spinner("Analyzing methodology clarity..."):
        response = chain.run({"paper_text": text})
        st.write(response)
        try:
            return int(re.search(r'\d+', response).group())
        except:
            return 5 # Default score if parsing fails

def calculate_and_display_confidence(novelty_score, num_inconsistencies, clarity_score, citation_overlap):
    st.subheader("Review Confidence Score")
    
    # Formula: Start with a base of 80%
    # - Adjust for novelty (more novel = higher confidence)
    # - Penalize for inconsistencies
    # - Adjust based on methodology clarity
    # - Adjust for citation overlap
    
    novelty_bonus = (novelty_score * 10) # Max +10
    inconsistency_penalty = min(num_inconsistencies * 5, 20) # Max -20
    clarity_bonus = (clarity_score - 5) * 2 # Range -10 to +10
    citation_bonus = min(citation_overlap / 2, 10) # Max +10

    confidence = 80 + novelty_bonus - inconsistency_penalty + clarity_bonus + citation_bonus
    confidence = max(0, min(100, confidence)) # Clamp between 0 and 100

    st.metric(label="Review Confidence", value=f"{confidence:.1f}%", delta=f"{confidence-80:.1f} from base")
    return confidence


def generate_review(text, llm, novelty_score, num_inconsistencies):
    st.subheader("Generated Review")

    prompt_template = """
    You are a world-class reviewer for a top-tier conference like NeurIPS or IEEE. You have been asked to review the following research paper.

    Here is the full text of the paper:
    ---
    {paper_text}
    ---

    Based on my initial analysis, I have the following context:
    - The novelty score is {novelty_score:.2f} (where a higher score is better, closer to 1.0 is highly novel).
    - I found {num_inconsistencies} potential statistical inconsistencies.

    Please provide a comprehensive review of the paper, including the following sections:
    1.  **Summary:** Briefly summarize the paper's key contributions.
    2.  **Strengths:** What are the main strengths of this paper?
    3.  **Weaknesses:** What are the main weaknesses? Please comment on the novelty and any potential issues with methodology or statistical claims.
    4.  **Suggestions for Improvement:** What specific experiments or revisions would improve this paper?
    5.  **Overall Recommendation:** Your final recommendation (e.g., Accept, Leaning Accept, Leaning Reject, Reject).

    Your review should be constructive, critical, and well-supported by evidence from the paper.
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["paper_text", "novelty_score", "num_inconsistencies"])
    chain = LLMChain(llm=llm, prompt=prompt)

    with st.spinner("Generating review..."):
        review = chain.run({
            "paper_text": text,
            "novelty_score": novelty_score,
            "num_inconsistencies": num_inconsistencies
        })
        st.markdown(review)
    return review

# --- Main App ---
with st.sidebar:
    st.header("Credentials")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password")

if not openrouter_api_key:
    st.info("Please enter your OpenRouter API key to proceed.")
    st.stop()

llm = get_llm(openrouter_api_key)
sentence_model = get_sentence_transformer()

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type="pdf")

if uploaded_file:
    if st.button("Analyze Paper"):
        st.info("Analyzing... this may take a few moments.")
        
        paper_text = extract_text_from_pdf(uploaded_file)
        title, abstract = get_title_and_abstract(paper_text)
        
        st.header("Uploaded Paper")
        st.write(f"**Title:** {title}")
        st.write(f"**Abstract:** {abstract}")

        novelty_score, citation_overlap = check_novelty(title, abstract, sentence_model)
        num_inconsistencies = detect_statistical_inconsistencies(paper_text, llm)
        clarity_score = get_methodology_clarity(paper_text, llm)
        
        calculate_and_display_confidence(novelty_score, num_inconsistencies, clarity_score, citation_overlap)

        generate_review(paper_text, llm, novelty_score, num_inconsistencies)

        st.success("Analysis complete!")
