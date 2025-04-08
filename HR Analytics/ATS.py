from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import io
import os 
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        # Take the first page for simplicity, or loop through images for all pages
        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No File uploaded")

# Streamlit App Configuration
st.set_page_config(page_title="ATS Expert", layout="wide")

# Apply custom CSS for a more professional UI design
st.markdown(
    """
    <style>
    .main {
        background-color: #F8F9FA;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #4B4F58;
        font-family: 'Arial', sans-serif;
        font-size: 2.8rem;
        font-weight: 600;
        margin-bottom: 20px;
    }
    h2 {
        color: #4B4F58;
        font-family: 'Arial', sans-serif;
        font-size: 1.8rem;
        font-weight: 500;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 12px 25px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 8px;
        transition: background-color 0.3s ease;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextArea>textarea {
        padding: 12px;
        font-size: 16px;
        width: 100%;
        border-radius: 8px;
        border: 1px solid #BDC3C7;
        margin-top: 15px;
    }
    .stFileUploader>div {
        border: 2px dashed #007BFF;
        padding: 25px;
        border-radius: 15px;
        background-color: #F9FBFD;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stFileUploader>div:hover {
        background-color: #e6f0ff;
    }
    .stWarning {
        color: #E74C3C;
        font-weight: bold;
    }
    .stSuccess {
        color: #2ECC71;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #28A745;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 8px;
        margin-top: 20px;
    }
    .stDownloadButton>button:hover {
        background-color: #218838;
    }

    /* Custom divider style */
    .divider {
        border-left: 2px solid #D1D1D1;
        height: 100%;
        margin-left: 20px;
        margin-right: 20px;
    }

    /* Left column styling */
    .left-column {
        background-color: #F4F6F9;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        height: 100%;
    }

    /* Right column styling */
    .right-column {
        padding: 20px;
        border-radius: 15px;
    }

    </style>
    """, unsafe_allow_html=True)

# Header and Introduction
st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
        <h1>AI-Powered Resume Analyzer</h1>
    </div>
""", unsafe_allow_html=True)

# Create UI Layout with a clear division between left (buttons) and right (inputs)
col1, col2 = st.columns([1, 3])  # Left column (buttons) | Right column (input fields)

# Left Column (Buttons)
with col1:
    st.header("Actions")
    st.write("Choose an action to perform with your uploaded resume.")
    submit1 = st.button("Analyze Resume")
    submit2 = st.button("Generate Cover Letter")
    submit3 = st.button("Skill Improvement Suggestions")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)  # Divider line after buttons

# Right Column (Job Description & Resume Upload)
with col2:
    st.markdown('<div class="right-column">', unsafe_allow_html=True)
    st.header("Upload Resume & Job Description")
    input_text = st.text_area("Enter Job Description:", key="input", height=150)
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

    if uploaded_file is not None:
        st.success("Resume uploaded successfully!")
    st.markdown('</div>', unsafe_allow_html=True)

# Define Prompts for different actions
input_prompt1 = """
You are an experienced Technical HR Manager. Evaluate the resume for:
- Role Fit: Data Analyst, Data Engineer, Data Scientist?
- Strengths & Weaknesses
- Seniority Level: Associate, Mid-Senior, or Senior?
- Technical Skills: Python, SQL, Spark, Cloud, ML, BI tools
- Domain Expertise: Healthcare, Finance, Retail?
"""

input_prompt2 = """
You are a recruiter helping generate a tailored cover letter. The cover letter should include:
- Introduction: Interest in the company and role.
- Key Skills & Qualifications: Relevant to job description.
- Experience & Achievements: Matching job responsibilities.
- Conclusion: Enthusiasm for the role, willingness for discussion.
"""

input_prompt3 = """
You are an ATS scanner. Evaluate the resume and determine:
- Percentage Match with the job description.
- Missing Keywords.
- Improvement Suggestions.
"""

# Handle Button Clicks
if submit1:
    if uploaded_file is not None and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("Resume Analysis")
        st.write(response)
    else:
        st.warning("Please upload a resume and enter the job description.")

elif submit2:
    if uploaded_file is not None and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt2, pdf_content, input_text)
        st.subheader("Generated Cover Letter")
        st.write(response)
        
        # Convert into downloadable file
        cover_letter_bytes = response.encode("utf-8")
        st.download_button(
            label="Download Cover Letter",
            data=cover_letter_bytes,
            file_name="Cover_Letter.doc",
            mime="text/plain"
        )
    else:
        st.warning("Please upload a resume and enter the job description.")

elif submit3:
    if uploaded_file is not None and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("Skill Improvement Suggestions")
        st.write(response)  
    else:
        st.warning("Please upload a resume and enter the job description.") 
