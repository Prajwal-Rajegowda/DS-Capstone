import streamlit as st
from utils import init_models
from utils import repo_loader
from utils import vector_db
from utils import generator as readme_generator

# Set up the page layout and title
st.set_page_config(page_title="AI README Generator", page_icon="📄", layout="centered")

st.title("📄 AI-Powered README Generator")
st.write("Automatically generate a comprehensive `README.md` for any GitHub repository using Gemini and RAG!")

if 'is_repo_cloned' not in st.session_state:
    st.session_state.is_repo_cloned = False

# Input field for the GitHub Repository URL
repo_url = st.text_input(
    "Enter GitHub Repository URL:", 
    placeholder="https://github.com/Prajwal-Rajegowda/Cafeteria",
    on_change=lambda: st.session_state.update({"is_repo_cloned" : False})
)

# Button to trigger the pipeline
if st.button("Generate README", type="primary"):
    if not repo_url.strip():
        st.error("Please enter a valid GitHub repository URL.")
    else:
        try:
            # Use Streamlit's status container to show progress step-by-step
            with st.status("Processing Repository... This may take a few minutes.", expanded=True) as status:
                
                st.write("⚙️ Initializing models...")
                models = init_models.init_models()

                st.write(f"📥 Cloning repository `{repo_url}`...")
                loader = repo_loader.RepoLoader()
                if(st.session_state.is_repo_cloned == False):
                   loader.clone_repo(repo_url)
                   st.session_state.update({"is_repo_cloned" : True})
                repo_files = loader.get_code_files()

                if not repo_files:
                    status.update(label="Failed to find valid code files.", state="error")
                    st.error("No valid code files found in the provided repository.")
                    st.stop()

                st.write(f"🧠 Processing {len(repo_files)} files and generating vector embeddings...")
                db = vector_db.RAGDatabase(models_instance=models)
                db.process_and_store(repo_files)

                st.write("✍️ Generating README sections...")
                generator = readme_generator.ReadmeGenerator(models_instance=models, db_collection=db.collection)
                output_filename = "GENERATED_README.md"
                generator.generate_full_readme(output_filename=output_filename)
                
                status.update(label="README Generation Complete!", state="complete", expanded=False)

            # Once complete, read the generated markdown file
            with open(output_filename, "r", encoding="utf-8") as f:
                readme_content = f.read()
            
            st.success("Successfully generated README!")
            
            # Provide a download button for the user to download the generated file
            st.download_button(
                label="⬇️ Download README.md",
                data=readme_content,
                file_name="README.md",
                mime="text/markdown"
            )

            # Display a preview of the generated README
            st.markdown("---")
            st.markdown("### Preview")
            st.markdown(readme_content)

        except Exception as e:
            st.error(f"An error occurred during generation: {str(e)}")