class ReadmeGenerator:
    def __init__(self, models_instance, db_collection):
        self.models = models_instance
        self.collection = db_collection

    def retrieve_context(self, query, top_k=5):
        query_embedding = self.models.get_gemini_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = results['documents'][0]
        context = "\n\n--- Code Snippet ---\n\n".join(documents)
        return context

    def generate_section(self, section_name, specific_instruction):
        print(f"Generating section: {section_name}...")

        context = self.retrieve_context(specific_instruction)

        prompt = f"""
        You are an expert technical writer and software engineer.
        Your task is to write the '{section_name}' section of a README.md file.
        
        Based ONLY on the following retrieved code snippets from the repository, follow this instruction:
        {specific_instruction}
        
        Repository Context:
        {context}
        
        Return ONLY the Markdown content for this section. Do not include introductory or concluding conversational text.
        """
        
        # return self.models.gemini_model("gemini-2.5-flash", prompt)
        return self.models.get_nvidia_api_response(prompt)

    def generate_full_readme(self, output_filename="GENERATED_README.md"):

        sections = {
            "Project Title & Summary": "Determine the likely name of this project and write a 2-paragraph summary explaining its primary purpose and what the code does.",
            "Features": "List the core features and functionalities of this codebase as bullet points.",
            "Architecture & Technologies": "Describe the high-level architecture, main components, and any specific libraries or frameworks used.",
            "Getting Started": "Provide instructions on how to run or initialize this project, including any entry point files (like main.py)."
        }
        
        full_readme = ""
        for title, instruction in sections.items():
            content = self.generate_section(title, instruction)
            full_readme += f"## {title}\n\n{content}\n\n---\n\n"
            
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(full_readme)
            
        print(f"\nREADME generation complete! Saved to {output_filename}")