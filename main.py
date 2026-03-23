from utils import init_models
from utils import repo_loader
from utils import vector_db
from utils import generator as readme_generator

def main():
    print("Initializing models...")
    models = init_models.init_models()

    loader = repo_loader.RepoLoader()
    test_repo_url = "https://github.com/abhi270502/Capstone-Project.git"
    loader.clone_repo(test_repo_url)
    repo_files = loader.get_code_files()

    print("\nProcessing and embedding codebase...")
    db = vector_db.RAGDatabase(models_instance=models)
    db.process_and_store(repo_files)

    print("\nStarting README generation...")
    generator = readme_generator.ReadmeGenerator(models_instance=models, db_collection=db.collection)
    generator.generate_full_readme()

if __name__ == "__main__":
    main()