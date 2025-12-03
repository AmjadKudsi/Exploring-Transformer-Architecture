# extract embeddings for the word "bank" in two different sentences to see how BERT creates different representations based on context

from transformers import AutoTokenizer, AutoModel
import torch

def bert_contextualized_embeddings():
    """Extract BERT's contextualized embeddings"""
    print("BERT Contextualized Embeddings:")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    # Test word in different contexts
    contexts = [
        "The bank of the river is muddy.",
        "I need to go to the bank to withdraw money."
    ]
    
    for i, context in enumerate(contexts):
        inputs = tokenizer(context, return_tensors="pt")
        tokens = tokenizer.tokenize(context)
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # TODO: Find the position of 'bank' token in the tokens list
        bank_position = None
        for j, token in enumerate(tokens):
            if token == 'bank':
                bank_position = j + 1
                break
        
        if bank_position is not None:
            # TODO: Extract the embedding for the 'bank' token from hidden_states
            embedding = hidden_states[0, bank_position, :]
            
            print(f"Context {i+1}: '{context}'")
            print(f"  'bank' token embedding shape: {embedding.shape}")
            print(f"  'bank' embedding norm: {embedding.norm().item():.3f}")
            print()

if __name__ == "__main__":
    bert_contextualized_embeddings()