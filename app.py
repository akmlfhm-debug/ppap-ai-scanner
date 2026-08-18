import streamlit as st
from google import genai
import pandas as pd
import json
import os
import tempfile
import time

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="AI PPAP Scanner")
st.title("🤖 AI-Powered PPAP Document Scanner")
st.write("Upload your engineering PDFs. The AI will scan the contents and generate an automated PPAP Checklist.")

# Securely input API Key
api_key = st.text_input("Enter your Google Gemini API Key:", type="password")

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload PPAP Documents (PDFs only)", 
    type=['pdf'], 
    accept_multiple_files=True
)

if st.button("Scan Documents & Generate Checklist"):
    if not api_key:
        st.error("Please enter an API Key to proceed.")
        st.stop()
        
    if not uploaded_files:
        st.error("Please upload at least one PDF document.")
        st.stop()

    # Initialize the new SDK Client
    client = genai.Client(api_key=api_key)
    
    with st.spinner("Scanning documents with AI... This usually takes 15-30 seconds."):
        try:
            # 1. Save uploaded files to temporary storage so the API can read them
            gemini_files = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                
                # Upload to Gemini API
                uploaded_to_gemini = client.files.upload(
                    file=tmp_path, 
                    config={'display_name': file.name}
                )
                gemini_files.append(uploaded_to_gemini)
                os.unlink(tmp_path) # Clean up temp file

            # 2. Instruct the AI on exactly what to look for and how to output it
            prompt = """
            You are a Senior Supplier Quality Engineer. Review the attached PDF documents.
            Your job is to identify if the required PPAP documents are present and if they meet standard passing criteria.
            
            Evaluate the following items:
            1. Design Record / Drawing
            2. Process Flow Diagram (PFD)
            3. Process FMEA (PFMEA)
            4. Control Plan
            5. First Article Inspection (FAI)
            6. Process Capability (CPK)
            
            Return the results STRICTLY as a JSON array of objects. Do not include markdown formatting like ```json.
            Each object must have the following keys:
            - "Document": The name of the required document category.
            - "Status": "OK" if present and passing, "MISSING" if not found, "ALERT" if found but has issues.
            - "Findings": A brief, accurate explanation of what you found (e.g., "CPK is > 1.33", "RPN is 192", or "Not provided").
            """

            contents = gemini_files + [prompt]
            
            # 3. Send the PDFs and the Prompt to the AI with a Retry Loop
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    # Switched to the Flash model for faster processing and fewer 503 errors
                    response = client.models.generate_content(
                        model='gemini-1.5-flash', 
                        contents=contents
                    )
                    break # Success! Break out of the retry loop
                except Exception as e:
                    if "503" in str(e) and attempt < max_retries - 1:
                        time.sleep(3) # Wait 3 seconds and try again
                        continue
                    else:
                        raise e # If it is not a 503 or we ran out of retries, throw the error
            
            # Clean up files from Gemini API storage
            for f in gemini_files:
                client.files.delete(name=f.name)

            # 4. Parse the AI's JSON response
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3] 
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3]
                
            checklist_data = json.loads(raw_text)
            df = pd.DataFrame(checklist_data)

            # 5. Display the results with color coding
            st.success("Scanning Complete!")
            
            def color_status(val):
                if val == 'OK':
                    return 'background-color: #c6efce; color: #006100; font-weight: bold;'
                elif val == 'MISSING':
                    return 'background-color: #ffc7ce; color: #9c0006; font-weight: bold;'
                elif val == 'ALERT':
                    return 'background-color: #ffeb9c; color: #9c5700; font-weight: bold;'
                return ''

            styled_df = df.style.map(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)

        except json.JSONDecodeError:
            st.error("The AI failed to format the response as JSON. Please try scanning again.")
        except Exception as e:
            st.error(f"An error occurred during scanning: {e}")
