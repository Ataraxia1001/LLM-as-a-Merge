from openai import OpenAI

# Create an OpenAI client configured for the Ollama local server
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # Required but ignored by Ollama
)

def ollama_infer(prompt: str, model: str = "llama3.2") -> str:
    """Run a single inference call using the Ollama local server."""
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=model,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


if __name__ == "__main__":
    model_name = "llama3.2"  # Ollama model name
    user_prompt = "Explain what inference means in one short paragraph."

    try:
        output = ollama_infer(user_prompt, model=model_name)
        print("Prompt:", user_prompt)
        print("\nModel response:\n")
        print(output)
    except RuntimeError as err:
        print(f"Error: {err}")
