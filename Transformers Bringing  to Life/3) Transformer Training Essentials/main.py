import torch
from torch.utils.data import DataLoader

from transformer.model import Transformer
from data_utils import create_synthetic_data, Vocabulary, TranslationDataset
from train import TransformerTrainer

def collate_fn(batch):
    """Custom collate function"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    tgt_output_batch = [item['tgt_output'] for item in batch]
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    tgt_output_batch = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, batch_first=True, padding_value=0)
    
    return {'src': src_batch, 'tgt': tgt_batch, 'tgt_output': tgt_output_batch}

def train_transformer():
    """Train Transformer model on synthetic data"""
    print("Training Transformer Model...")
    
    # Create data
    src_sentences, tgt_sentences = create_synthetic_data(num_samples=200)
    
    # Build vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    print(f"Source vocab size: {src_vocab.size}")
    print(f"Target vocab size: {tgt_vocab.size}")
    
    # Create dataset and dataloader
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=15)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Create model
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TransformerTrainer(model, train_loader, lr=1e-3, warmup_steps=25)
    print("Trainer created successfully!")
    
    return model, src_vocab, tgt_vocab

def main():
    model, src_vocab, tgt_vocab = train_transformer()
    print("Setup completed successfully!")

if __name__ == "__main__":
    main()