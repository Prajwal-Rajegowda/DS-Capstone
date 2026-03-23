import os
import shutil
from git import Repo

class RepoLoader:
    def __init__(self, clone_dir="./temp_repo"):
        self.clone_dir = clone_dir

    def clone_repo(self, repo_url):
        if os.path.exists(self.clone_dir):
            shutil.rmtree(self.clone_dir)
        
        print(f"Cloning {repo_url}...")
        Repo.clone_from(repo_url, self.clone_dir)
        print("Cloning complete.")

    def get_code_files(self, allowed_extensions=('.py', '.md', '.txt')):
        documents = []
        for root, _, files in os.walk(self.clone_dir):
            if '.git' in root:
                continue
            
            for file in files:
                if file.endswith(allowed_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            documents.append({
                                "path": file_path, 
                                "content": f.read()
                            })
                    except Exception as e:
                        print(f"Could not read {file_path}: {e}")
        return documents