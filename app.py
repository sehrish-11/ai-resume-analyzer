import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("AI Resume Analyzer 📝")

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        text = extract_text(uploaded_file)  # FIXED: directly use extract_text
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Function to calculate similarity
def calculate_similarity(resume_text, job_desc):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_emb = model.encode([resume_text])
    jd_emb = model.encode([job_desc])
    similarity = cosine_similarity(resume_emb, jd_emb)[0][0]
    return similarity

# Upload resume
resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description here:")

if resume_file and job_desc:
    resume_text = extract_pdf_text(resume_file)
    if resume_text:
        score = calculate_similarity(resume_text, job_desc)
        st.subheader(f"Similarity Score: {score:.2f}")

        if score > 0.75:
            st.success("✅ Strong Match! Your resume aligns well with the JD.")
        elif score > 0.50:
            st.warning("⚠️ Medium Match. Consider improving your resume.")
        else:
            st.error("❌ Weak Match. Resume doesn’t align much with the JD.")
