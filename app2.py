import streamlit as st
import pandas as pd
import re
import fitz
from docx import Document
from io import BytesIO
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from groq import Groq
from dotenv import load_dotenv

# ==================================================
# ENV & CLIENT
# ==================================================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="AI Resume Job Matcher",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume & Job Matching Platform")
st.caption("Resume Matching • Skill Gaps • AI Feedback • PDF Report")

# ==================================================
# TEXT CLEANING
# ==================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==================================================
# RESUME EXTRACTION
# ==================================================
@st.cache_data
def extract_resume_text(file_bytes, filename):
    if filename.endswith(".pdf"):
        text = ""
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf:
            text += page.get_text()
        return text

    if filename.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

# ==================================================
# EXPERIENCE PARSER
# ==================================================
def extract_years(text):
    match = re.search(r"(\d+)\+?\s*(years|yrs)", text.lower())
    return int(match.group(1)) if match else 0

# ==================================================
# LOAD DATASETS
# ==================================================
jobs_df = pd.read_csv("NaukriData_Data Science.csv")
skills_df = pd.read_csv("job_titles_classification_extended.csv")

jobs_df.columns = jobs_df.columns.str.strip().str.lower()

def find_col(keys):
    for c in jobs_df.columns:
        for k in keys:
            if k in c:
                return c
    st.error(f"❌ Missing column: {keys}")
    st.stop()

desc_col = find_col(["skills", "description"])
title_col = find_col(["title"])
company_col = find_col(["company"])
salary_col = find_col(["salary", "package"])
location_col = find_col(["location"])
exp_col = find_col(["experience"])
url_col = find_col(["url", "link"])

jobs_df["clean_job"] = jobs_df[desc_col].astype(str).apply(clean_text)
jobs_df["jobtitle"] = jobs_df[title_col]
jobs_df["company"] = jobs_df[company_col]
jobs_df["salary"] = jobs_df[salary_col]
jobs_df["location"] = jobs_df[location_col].astype(str).str.title()
jobs_df["experience"] = jobs_df[exp_col].astype(str)
jobs_df["exp_years"] = jobs_df["experience"].apply(extract_years)
jobs_df["apply_url"] = jobs_df[url_col].astype(str)

# ==================================================
# TF-IDF MATCHING
# ==================================================
@st.cache_resource
def build_vectorizer(texts):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(texts)
    return tfidf, vectors

tfidf, job_vectors = build_vectorizer(jobs_df["clean_job"].tolist())

def match_jobs(resume_text, df):
    vec = tfidf.transform([resume_text])
    sim = cosine_similarity(vec, job_vectors[df.index]).flatten()
    out = df.copy()
    out["match_percent"] = (sim * 100).round(2)
    return out.sort_values("match_percent", ascending=False)

# ==================================================
# SKILLS
# ==================================================
skill_list = sorted(set(skills_df.astype(str).values.flatten()))

def extract_skills(text):
    text = text.lower()
    return sorted([s.lower() for s in skill_list if s.lower() in text])

def skill_gap(resume_text, job_text):
    r = set(extract_skills(resume_text))
    j = set(extract_skills(job_text))
    return sorted(r & j), sorted(j - r)

# ==================================================
# RESUME SCORE
# ==================================================
def resume_score(resume_text):
    score = 0
    score += min(len(extract_skills(resume_text)) / 15, 1) * 40
    score += min(extract_years(resume_text) / 5, 1) * 25
    score += min(len(resume_text.split()) / 500, 1) * 15
    score += min(resume_text.count("project") / 3, 1) * 20
    return round(score, 2)

# ==================================================
# LLM FEEDBACK (UPDATED MODEL ✅)
# ==================================================
def llm_feedback(resume_text, score, missing, jobs):
    job_list = "\n".join(
        f"{r['jobtitle']} at {r['company']}" for _, r in jobs.iterrows()
    )

    prompt = f"""
Resume Score: {score}/100
Missing Skills: {", ".join(missing[:8])}

Resume:
{resume_text[:3000]}

Top Jobs:
{job_list}

Give:
• Strengths
• Weaknesses
• Skills to learn
• Resume improvement tips
• Project ideas
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

# ==================================================
# PDF REPORT
# ==================================================
def generate_pdf(score, feedback):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "AI Resume Evaluation Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Resume Score: {score}/100")

    text = c.beginText(50, 740)
    text.setFont("Helvetica", 11)
    for line in feedback.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.save()
    buffer.seek(0)
    return buffer

# ==================================================
# SIDEBAR (ADMIN)
# ==================================================
with st.sidebar:
    st.header("📊 Admin Dashboard")
    st.metric("Total Jobs", len(jobs_df))
    st.metric("Companies", jobs_df["company"].nunique())
    st.metric("Locations", jobs_df["location"].nunique())

# ==================================================
# UI
# ==================================================
uploaded = st.file_uploader("📄 Upload Resume (PDF / DOCX)", ["pdf", "docx"])
min_exp = st.slider("🎓 Minimum Experience (Years)", 0, 15, 0)

if uploaded:
    with st.spinner("Analyzing resume..."):
        resume_text = extract_resume_text(uploaded.getvalue(), uploaded.name)
        resume_clean = clean_text(resume_text)

        filtered = jobs_df[jobs_df["exp_years"] >= min_exp]
        matched = match_jobs(resume_clean, filtered)

        score = resume_score(resume_clean)
        _, missing = skill_gap(resume_clean, " ".join(matched.head(3)["clean_job"]))

        feedback = llm_feedback(resume_text, score, missing, matched.head(3))

    st.metric("📊 Resume Score", f"{score} / 100")

    st.subheader("🧠 AI Feedback")
    st.write(feedback)

    pdf = generate_pdf(score, feedback)
    st.download_button("📄 Download PDF Report", pdf, "resume_report.pdf")

    st.subheader("🔥 Top Job Matches")

    for _, r in matched.head(10).iterrows():
        m, ms = skill_gap(resume_clean, r["clean_job"])
        st.markdown(
            f"""
            <div style="padding:16px;border-radius:12px;background:#1e1e1e;margin-bottom:12px">
            <h4>{r['jobtitle']}</h4>
            <b>{r['company']}</b><br>
            📍 {r['location']} | 💰 {r['salary']}<br>
            🔥 Match: {r['match_percent']}%<br>
            ✅ Skills: {", ".join(m[:8]) if m else "—"}<br>
            ❌ Missing: {", ".join(ms[:8]) if ms else "None"}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f"[🚀 Apply Now]({r['apply_url']})")
