import streamlit as st
from utils import init_models
from utils import repo_loader
from utils import vector_db
from utils import generator as readme_generator

st.set_page_config(page_title="AI README Generator", page_icon="📄", layout="wide")

if 'is_repo_cloned' not in st.session_state:
    st.session_state.is_repo_cloned = False
if 'generated_readme' not in st.session_state:
    st.session_state.generated_readme = ""

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Customize your README output.")
    
    st.subheader("Sections to Generate")
    include_summary = st.checkbox("Project Summary", value=True)
    include_features = st.checkbox("Features", value=True)
    include_arch = st.checkbox("Architecture & Tech", value=True)
    include_getting_started = st.checkbox("Getting Started", value=True)
    include_contributing = st.checkbox("Contributing Guidelines", value=False)
    
    st.divider()

    if st.button("Reset App & Clear Cache"):
        st.session_state.clear()
        st.rerun()

# --- MAIN UI ---
st.title("AI-Powered README Generator")
st.write("Automatically generate a comprehensive `README.md` for any GitHub repository using RAG!")

repo_url = st.text_input(
    "Enter GitHub Repository URL:", 
    placeholder="https://github.com/Prajwal-Rajegowda/Cafeteria",
    on_change=lambda: st.session_state.update({"is_repo_cloned" : False, "generated_readme": ""})
)

# Button to trigger the pipeline
if st.button("Generate README", type="primary"):
    if not repo_url.strip():
        st.error("Please enter a valid GitHub repository URL.")
    else:
        try:
            with st.status("Processing Repository... This may take a few minutes.", expanded=True) as status:
                
                st.write("Initializing models...")
                models = init_models.init_models()

                st.write(f"Cloning repository `{repo_url}`...")
                loader = repo_loader.RepoLoader()
                db = vector_db.RAGDatabase(models_instance=models)
                if not st.session_state.is_repo_cloned:
                    loader.clone_repo(repo_url)
                    st.session_state.is_repo_cloned = True
                    repo_files = loader.get_code_files()

                    if not repo_files:
                        status.update(label="Failed to find valid code files.", state="error")
                        st.error("No valid code files found in the provided repository.")
                        st.stop()

                    st.write(f"Processing {len(repo_files)} files and generating vector embeddings...")
                    db.process_and_store(repo_files)

                st.write("Generating README sections...")
                generator = readme_generator.ReadmeGenerator(models_instance=models, db_collection=db.collection)

                custom_sections = {}
                if include_summary: custom_sections["Project Title & Summary"] = "Determine the likely name of this project and write a 2-paragraph summary explaining its primary purpose."
                if include_features: custom_sections["Features"] = "List the core features and functionalities of this codebase as bullet points."
                if include_arch: custom_sections["Architecture & Technologies"] = "Describe the high-level architecture, main components, and any specific libraries or frameworks used."
                if include_getting_started: custom_sections["Getting Started"] = "Provide instructions on how to run or initialize this project."
                if include_contributing: custom_sections["Contributing"] = "Provide basic guidelines on how developers can contribute to this repository."
                
                output_filename = "GENERATED_README.md"
                generator.generate_full_readme(output_filename=output_filename, sections=custom_sections)
                
                status.update(label="README Generation Complete!", state="complete", expanded=False)

            with open(output_filename, "r", encoding="utf-8") as f:
                st.session_state.generated_readme = f.read()
            
            st.success("Successfully generated README!")

        except Exception as e:
            st.error(f"An error occurred during generation: {str(e)}")

# --- RESULTS UI ---
if st.session_state.generated_readme:
    st.markdown("---")
    st.header("✨ Your Generated README")
    
    # Create tabs for editing and previewing
    tab1, tab2 = st.tabs(["Edit Raw Markdown", "Rendered Preview"])
    
    with tab1:
        # Allow user to edit the generated markdown before downloading
        edited_readme = st.text_area(
            "Make manual tweaks here before downloading:", 
            value=st.session_state.generated_readme, 
            height=400
        )
        
        st.download_button(
            label="Download README.md",
            data=edited_readme,
            file_name="README.md",
            mime="text/markdown",
            use_container_width=True
        )
        
    with tab2:
        st.info("This is how your README will look when rendered on GitHub or PyPI.")
        st.markdown(edited_readme, unsafe_allow_html=True)