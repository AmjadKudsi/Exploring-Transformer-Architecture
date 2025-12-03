# complete the masked language modeling function by filling in the missing parts
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def bert_masked_language_modeling():
    """Demonstrate BERT's masked language modeling"""
    print("BERT Masked Language Modeling:")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    
    # TODO: Define test_sentences with [MASK] tokens
    test_sentences = [
        "The [MASK] is shining  brightly today.",
        "I love to [MASK] books in my free time."
    ]
    
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        # TODO: Find mask token and get top predictions
        # Hint: Use torch.where to find mask_token_index
        # Then get mask_logits and use torch.topk to get top 3 predictions
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
        mask_logits = predictions[0, mask_token_index, :]
        top_tokens = torch.topk(mask_logits, k=3, dim=1)
        
        print(f"Sentence: '{sentence}'")
        print("Top predictions:")
        # TODO: Loop through top predictions and print token and score
        for token_id, score in zip(top_tokens.indices[0], top_tokens.values[0]):
            token = tokenizer.decode([token_id])
            print(f"  '{token.strip()}': score={score.item():.2f}")
        print()

if __name__ == "__main__":
    bert_masked_language_modeling()
