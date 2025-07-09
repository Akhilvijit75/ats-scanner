import streamlit as st
import requests
import json
import pdfplumber
import docx
from PIL import Image
import pytesseract
from io import BytesIO
import pandas as pd
import os
import datetime
import re
from collections import Counter

# Page configuration
st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

# API Keys
GEMINI_API_KEY = "AIzaSyAdPsOIVPj1NVZOveB05AZwSt9eoSx0_d0"
CSV_FILE = "ats_results.csv"

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def init_csv():
    """Initialize CSV file if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=[
            'timestamp', 'filename', 'ats_score', 'github_username', 
            'github_repos', 'github_followers', 'strengths', 'improvements'
        ])
        df.to_csv(CSV_FILE, index=False)

def save_results(filename, ats_score, github_data, strengths, improvements):
    """Save analysis results to CSV, avoiding duplicates based on GitHub username"""
    try:
        init_csv()
        df = pd.read_csv(CSV_FILE)
        
        github_username = github_data.get('username', '').strip()
        
        # Check for duplicates based on GitHub username (if provided)
        if github_username:
            # Remove any existing entries with the same GitHub username
            df = df[df['github_username'] != github_username]
            st.info(f"ℹ️ Updated existing record for GitHub user: {github_username}")
        
        new_data = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'ats_score': ats_score,
            'github_username': github_username,
            'github_repos': github_data.get('repos', 0),
            'github_followers': github_data.get('followers', 0),
            'strengths': '; '.join(strengths),
            'improvements': '; '.join(improvements)
        }
        
        new_df = pd.DataFrame([new_data])
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(BytesIO(file.read()))
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return ' '.join(text)
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_image(file):
    """Extract text from image using OCR"""
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error reading image: {str(e)}")
        return ""

def extract_github_username(text):
    """Extract GitHub username from resume text"""
    patterns = [
        r'github\.com/([a-zA-Z0-9_-]+)',
        r'github\.io/([^/\s]+)',
        r'GitHub:\s*@?([a-zA-Z0-9_-]+)',
        r'[Gg]itHub\s*:?\s*([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

def fetch_github_profile(username):
    """Fetch GitHub profile data"""
    try:
        url = f"https://api.github.com/users/{username}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"GitHub API error: {str(e)}")
        return None

def fetch_github_repos(username):
    """Fetch user repositories"""
    try:
        url = f"https://api.github.com/users/{username}/repos?per_page=50"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching repos: {str(e)}")
        return []

def analyze_languages(repos):
    """Analyze programming languages from repositories"""
    languages = [repo.get('language') for repo in repos if repo.get('language')]
    return Counter(languages).most_common(5)

def analyze_resume_with_ai(resume_text, job_description=None):
    """Analyze resume using Gemini AI"""
    if not job_description:
        job_description = "General software engineering position requiring programming skills, experience with databases, and good communication abilities."
    
    prompt = f"""
    You are an expert ATS (Applicant Tracking System) analyzer. Carefully analyze this resume and provide a detailed, objective assessment.

    SCORING CRITERIA (be strict and realistic):
    - 90-100: Exceptional resume with all key elements, perfect formatting, strong experience
    - 80-89: Very good resume with most elements present, good formatting, relevant experience  
    - 70-79: Good resume with some elements missing, decent formatting, adequate experience
    - 60-69: Average resume with several gaps, basic formatting, limited experience
    - 50-59: Below average resume with many issues, poor formatting, weak experience
    - Below 50: Poor resume with major problems, very poor formatting, no relevant experience

    ANALYZE THESE SPECIFIC ASPECTS:
    1. Does the resume contain relevant keywords from the job description?
    2. Are technical skills clearly listed and relevant?
    3. Is the formatting clean and ATS-friendly?
    4. Are achievements quantified with numbers/metrics?
    5. Is contact information complete?
    6. Does experience match the job requirements?
    7. Is the resume length appropriate (1-2 pages)?
    8. Are there any spelling/grammar errors?

    Provide your response in JSON format:
    {{
        "score": [realistic score 0-100 based on actual resume quality],
        "strengths": [list exactly 3 specific strengths found in this resume],
        "improvements": [list exactly 3 specific areas this resume needs to improve],
        "technical_skills": [list all technical skills/technologies mentioned],
        "missing_keywords": [list 3-5 important keywords from job description that are missing]
    }}

    JOB DESCRIPTION:
    {job_description}

    RESUME CONTENT TO ANALYZE:
    {resume_text}

    Be honest and critical in your assessment. Base the score strictly on the resume quality, not on being encouraging. Return only valid JSON.
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            data = response.json()
            text_response = data['candidates'][0]['content']['parts'][0]['text']
            
            # Clean up JSON response
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0].strip()
            elif "```" in text_response:
                text_response = text_response.split("```")[1].split("```")[0].strip()
            
            return json.loads(text_response)
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def admin_login():
    """Admin login page"""
    st.title("🔐 Admin Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.current_page = "admin"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    if st.button("← Back to Main"):
        st.session_state.current_page = "main"
        st.rerun()

def admin_dashboard():
    """Admin dashboard"""
    st.title("📊 Admin Dashboard")
    
    # Load data
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            
            if not df.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Resumes", len(df))
                col2.metric("Average ATS Score", f"{df['ats_score'].mean():.1f}")
                col3.metric("Highest Score", df['ats_score'].max())
                col4.metric("With GitHub", len(df[df['github_username'] != '']))
                
                # Recent submissions
                st.subheader("Recent Submissions")
                display_df = df[['timestamp', 'filename', 'ats_score', 'github_username']].tail(10)
                st.dataframe(display_df, use_container_width=True)
                
                # Download data
                if st.button("📥 Download All Data"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "ats_analysis_data.csv",
                        "text/csv"
                    )
            else:
                st.info("No data available yet.")
        else:
            st.info("No data file found.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    
    # Logout
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.current_page = "main"
        st.rerun()

def main_app():
    """Main application"""
    st.title("📄 ATS Resume Checker")
    st.write("Upload your resume to check ATS compatibility and analyze your GitHub profile.")
    
    # Admin login button in sidebar
    with st.sidebar:
        if st.button("🔐 Admin Login"):
            st.session_state.current_page = "login"
            st.rerun()
    
    # File upload
    uploaded_file = st.file_uploader("Choose your resume", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'])
    
    # Optional inputs
    col1, col2 = st.columns(2)
    with col1:
        job_description = st.text_area("Job Description (Optional)", 
                                      placeholder="Paste job description for better analysis...")
    with col2:
        manual_github = st.text_input("GitHub Username (Optional)", 
                                     placeholder="Enter if not found in resume")
    
    if uploaded_file:
        # Extract text
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Reading your resume..."):
            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                uploaded_file.seek(0)
                resume_text = extract_text_from_docx(uploaded_file)
            elif file_extension in ['png', 'jpg', 'jpeg']:
                resume_text = extract_text_from_image(uploaded_file)
            else:
                st.error("Unsupported file format")
                resume_text = ""
        
        if resume_text.strip():
            # Extract GitHub username from resume or use manual input
            extracted_username = extract_github_username(resume_text)
            github_username = manual_github.strip() if manual_github.strip() else extracted_username
            
            # Show which username will be used
            if github_username:
                if manual_github.strip():
                    st.info(f"🔍 Using manually entered GitHub username: **{github_username}**")
                else:
                    st.info(f"🔍 Found GitHub username in resume: **{github_username}**")
            else:
                st.warning("⚠️ No GitHub username provided")
            
            if st.button("🔍 Analyze Resume", type="primary"):
                # First, always do ATS analysis
                with st.spinner("Analyzing your resume..."):
                    results = analyze_resume_with_ai(resume_text, job_description)
                
                if results:
                    # Create tabs for results
                    tab1, tab2 = st.tabs(["📋 ATS Analysis", "👨‍💻 GitHub Profile"])
                    
                    with tab1:
                        score = results.get('score', 0)
                        
                        # Display score
                        if score >= 80:
                            st.success(f"🎉 Excellent! ATS Score: {score}/100")
                        elif score >= 70:
                            st.info(f"👍 Good! ATS Score: {score}/100")
                        elif score >= 60:
                            st.warning(f"⚠️ Needs Improvement. ATS Score: {score}/100")
                        else:
                            st.error(f"❌ Poor. ATS Score: {score}/100")
                        
                        # Results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("✅ Strengths")
                            for strength in results.get('strengths', []):
                                st.write(f"• {strength}")
                            
                            st.subheader("🔧 Technical Skills")
                            for skill in results.get('technical_skills', []):
                                st.write(f"• {skill}")
                        
                        with col2:
                            st.subheader("📈 Improvements")
                            for improvement in results.get('improvements', []):
                                st.write(f"• {improvement}")
                            
                            st.subheader("🔑 Missing Keywords")
                            for keyword in results.get('missing_keywords', []):
                                st.write(f"• {keyword}")
                    
                    with tab2:
                        if github_username:
                            with st.spinner(f"Analyzing GitHub profile: {github_username}"):
                                profile = fetch_github_profile(github_username)
                                
                                if profile:
                                    # Profile header
                                    col1, col2 = st.columns([1, 3])
                                    
                                    with col1:
                                        if profile.get('avatar_url'):
                                            st.image(profile['avatar_url'], width=120)
                                    
                                    with col2:
                                        st.subheader(profile.get('name', 'No name'))
                                        st.write(f"@{profile.get('login')}")
                                        if profile.get('bio'):
                                            st.write(f"*{profile.get('bio')}*")
                                        
                                        # Metrics
                                        mcol1, mcol2, mcol3 = st.columns(3)
                                        mcol1.metric("Repositories", profile.get('public_repos', 0))
                                        mcol2.metric("Followers", profile.get('followers', 0))
                                        mcol3.metric("Following", profile.get('following', 0))
                                    
                                    # Repository analysis
                                    repos = fetch_github_repos(github_username)
                                    
                                    if repos:
                                        st.subheader("📊 Repository Analysis")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Top Languages:**")
                                            languages = analyze_languages(repos)
                                            for lang, count in languages:
                                                st.write(f"• {lang}: {count} repos")
                                        
                                        with col2:
                                            st.write("**Top Repositories:**")
                                            top_repos = sorted(repos, key=lambda x: x.get('stargazers_count', 0), reverse=True)[:5]
                                            for repo in top_repos:
                                                stars = repo.get('stargazers_count', 0)
                                                st.write(f"• [{repo.get('name')}]({repo.get('html_url')}) - ⭐ {stars}")
                                    
                                    # Save results with GitHub data
                                    github_data = {
                                        'username': github_username,
                                        'repos': profile.get('public_repos', 0),
                                        'followers': profile.get('followers', 0)
                                    }
                                    save_results(
                                        uploaded_file.name,
                                        results.get('score', 0),
                                        github_data,
                                        results.get('strengths', []),
                                        results.get('improvements', [])
                                    )
                                    st.success("✅ Analysis complete! Results saved.")
                                else:
                                    st.error(f"❌ GitHub user '{github_username}' not found")
                                    st.info("💡 Try checking the username spelling or make sure the profile is public")
                                    
                                    # Save results without GitHub data
                                    github_data = {'username': '', 'repos': 0, 'followers': 0}
                                    save_results(
                                        uploaded_file.name,
                                        results.get('score', 0),
                                        github_data,
                                        results.get('strengths', []),
                                        results.get('improvements', [])
                                    )
                                    st.success("✅ ATS analysis complete! Results saved.")
                        else:
                            st.info("🔍 No GitHub username provided")
                            st.write("To analyze your GitHub profile:")
                            st.write("• Add your GitHub URL to your resume, or")
                            st.write("• Enter your GitHub username in the field above")
                            
                            # Save results without GitHub data
                            github_data = {'username': '', 'repos': 0, 'followers': 0}
                            save_results(
                                uploaded_file.name,
                                results.get('score', 0),
                                github_data,
                                results.get('strengths', []),
                                results.get('improvements', [])
                            )
                            st.success("✅ ATS analysis complete! Results saved.")
                else:
                    st.error("❌ Failed to analyze resume. Please try again.")
        else:
            st.error("Could not extract text from file.")

# Main execution
def main():
    if st.session_state.current_page == "login":
        admin_login()
    elif st.session_state.current_page == "admin" and st.session_state.authenticated:
        admin_dashboard()
    else:
        main_app()

if __name__ == "__main__":
    main()