# modify the inference section to systematically test various beam widths and observe their impact

import torch
from torch.utils.data import DataLoader
import time

from transformer.model import Transformer
from data_utils import create_synthetic_data, Vocabulary, TranslationDataset
from train import TransformerTrainer
from inference import TransformerInference

def collate_fn(batch):
    """Custom collate function"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    tgt_output_batch = [item['tgt_output'] for item in batch]
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    tgt_output_batch = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, batch_first=True, padding_value=0)
    
    return {'src': src_batch, 'tgt': tgt_batch, 'tgt_output': tgt_output_batch}

def main():
    """Test inference with trained model"""
    print("Testing Transformer Inference...")
    
    # Prepare data and train model
    src_sentences, tgt_sentences = create_synthetic_data(num_samples=200)
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    # Create and train model
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=15)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    model = Transformer(
        src_vocab_size=src_vocab.size,
        tgt_vocab_size=tgt_vocab.size,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.1
    )
    
    # Quick training
    trainer = TransformerTrainer(model, train_loader, lr=1e-3, warmup_steps=25)
    print("Quick training for 2 epochs...")
    for epoch in range(2):
        avg_loss = trainer.train_epoch()
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
    # Test inference
    inference = TransformerInference(model, src_vocab, tgt_vocab)
    
    test_sentences = ["hello world", "good morning", "thank you very much"]
    
    print("\nInference Results:")
    # TODO: Create a list of beam widths to test (e.g., [1, 3, 5])
    b_widths = [1, 3, 5]
    for sentence in test_sentences:
        print(f"Source: '{sentence}'")
        print(f"Expected: '{' '.join(sentence.split()[::-1])}'")
        
        # TODO: Tests different beam widths
        for b in b_widths:
            start_time = time.time()
            greedy_result = inference.greedy_decode(sentence)
            greedy_time = time.time() - start_time
            
            start_time = time.time()
            beam_result = inference.beam_search(sentence, beam_width=b)
            beam_time = time.time() - start_time
            
            print(f"Greedy: '{greedy_result}' (time: {greedy_time:.3f}s)")
            print(f"Beam: '{beam_result}' (time: {beam_time:.3f}s)")
            print("-" * 50)
    

if __name__ == "__main__":
    main()