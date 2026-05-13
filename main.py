from ollama import Client


def ollama_infer(prompt: str, model: str = "llama3.2", host: str = "http://localhost:11434") -> str:
    """Run a single inference call against a local Ollama server."""
    try:
        client = Client(host=host)
        response = client.generate(model=model, prompt=prompt)
        return response.get("response", "")
    except Exception as exc:
        raise RuntimeError(
            "Failed to connect to Ollama. Ensure Ollama is installed and running, then run: "
            "'ollama pull llama3.2'."
        ) from exc


if __name__ == "__main__":
    model_name = "llama3.2"
    user_prompt = "Explain what inference means in one short paragraph."

    try:
        output = ollama_infer(user_prompt, model=model_name)
        print("Prompt:", user_prompt)
        print("\nModel response:\n")
        print(output)
    except Exception as err:
        print(f"Error: {err}")
