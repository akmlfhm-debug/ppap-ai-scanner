import streamlit as st
from google import genai
import pandas as pd
import json
import os
import tempfile
import fitz  # PyMuPDF library for PDF highlighting

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="AI Visual PPAP Validator")
st.title("👁️ AI Visual PPAP Validator")
st.write("This system uses AI to analyze documents, and deterministic logic to highlight the exact evidence on the PDFs.")

# Pull the API key securely from Streamlit Secrets (NO HARDCODING)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found! Please add GEMINI_API_KEY to your Streamlit secrets.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload PPAP Documents (PDFs only)", 
    type=['pdf'], 
    accept_multiple_files=True
)

if st.button("Run Detailed Visual Check"):
    if not uploaded_files:
        st.error("Please upload at least one PDF document.")
        st.stop()

    client = genai.Client(api_key=api_key)
    
    # Initialize these OUTSIDE the try block so they can be safely cleaned up
    gemini_files = []
    local_file_paths = [] 
    
    with st.spinner("Analyzing documents and rendering visual evidence (this takes ~30 seconds)..."):
        try:
            # 1. Save files locally for PyMuPDF, and upload to Gemini
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                    local_file_paths.append((file.name, tmp_path))
                    
                uploaded_to_gemini = client.files.upload(
                    file=tmp_path, 
                    config={'display_name': file.name}
                )
                gemini_files.append(uploaded_to_gemini)

            # 2. Instruct the AI to provide exact string matches for evidence
            prompt = """
            You are a rigorous Quality Inspector. Audit the attached PPAP documents (FAI, CPK, PFMEA, etc.).
            Identify specific failures, out-of-spec dimensions, or high RPNs (> 100).
            
            Return a STRICT JSON array where each object has:
            - "Document": Category (e.g., "FAI", "PFMEA")
            - "Status": "FAIL" or "ALERT"
            - "Finding": Explanation of the defect.
            - "Exact_Text": The EXACT string of text or number as it appears in the PDF that proves this finding (e.g., "65.661", "192"). This MUST be a verbatim match to the document. If nothing applies, return "None".
            """

            response = client.models.generate_content(
                model='gemini-1.5-pro',
                contents=gemini_files + [prompt]
            )
            
            # Clean up API storage
            for f in gemini_files:
                client.files.delete(name=f.name)

            # 3. Parse JSON Response
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3] 
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3]
                
            results = json.loads(raw_text)

            # 4. Display the Summary Table
            st.subheader("📊 Automated Audit Summary")
            df = pd.DataFrame(results)
            
            def color_status(val):
                if val == 'FAIL':
                    return 'background-color: #ffc7ce; color: #9c0006; font-weight: bold;'
                elif val == 'ALERT':
                    return 'background-color: #ffeb9c; color: #9c5700; font-weight: bold;'
                return ''
                
            st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)

            # 5. VISUAL EVIDENCE ENGINE (PyMuPDF)
            st.subheader("🔍 Visual Evidence Tracing")
            
            for item in results:
                evidence = str(item.get("Exact_Text", "None"))
                
                if evidence != "None" and evidence.strip() != "":
                    st.markdown(f"### Proving: {item['Finding']}")
                    st.write(f"**Target:** Searching PDFs for exactly `{evidence}`...")
                    
                    found_evidence = False
                    
                    # Scan every local PDF file
                    for fname, fpath in local_file_paths:
                        doc = fitz.open(fpath)
                        
                        # Scan every page
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            text_instances = page.search_for(evidence)
                            
                            if text_instances:
                                found_evidence = True
                                # Draw a bright rectangle over every instance found
                                for inst in text_instances:
                                    highlight = page.add_rect_annot(inst)
                                    if item['Status'] == 'FAIL':
                                        highlight.set_colors(stroke=(1, 0, 0)) # Red
                                    else:
                                        highlight.set_colors(stroke=(1, 0.64, 0)) # Orange
                                    highlight.set_border(width=3)
                                    highlight.update()
                                    
                                # Render the marked-up page as an image
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
                                img_bytes = pix.tobytes("png")
                                
                                st.image(img_bytes, caption=f"Evidence isolated in {fname} (Page {page_num + 1})")
                                break # Stop searching once we find the first proof image
                        
                        doc.close()
                        if found_evidence:
                            break
                            
                    if not found_evidence:
                        st.warning(f"AI identified '{evidence}', but the deterministic engine could not locate it visually.")
                        st.divider()

        except Exception as e:
            st.error(f"System Error: {e}")
            
        finally:
            # Always clean up local server temp files
            for _, fpath in local_file_paths:
                try:
                    os.unlink(fpath)
                except:
                    pass
