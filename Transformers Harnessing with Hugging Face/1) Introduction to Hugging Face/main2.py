# Create a text generation pipeline that can produce creative continuations of text prompts

from transformers import pipeline


def text_generation_demo():
    """Demonstrate text generation using Hugging Face pipeline"""
    print("Text Generation with Hugging Face Transformers")
    
    # TODO: Create a text generation pipeline using the "gpt2" model
    generator = pipeline("text-generation", model="gpt2")
    
    # Generate text from a prompt
    prompt = "The future of artificial intelligence is"
    # TODO: Use the pipeline to generate text from the prompt
    # Set max_length=50, num_return_sequences=2, truncation=True
    generated = generator(prompt,
            max_length=50,
            num_return_sequences=2,
            truncation=True)
    
    print(f"Original prompt: {prompt}")
    print("\nGenerated text:")
    for i, result in enumerate(generated):
        print(f"{i+1}. {result['generated_text']}")
    


def main():
    text_generation_demo()


if __name__ == "__main__":
    main()