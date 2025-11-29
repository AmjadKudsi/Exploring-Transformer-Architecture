# implement the entire main function from scratch
import torch
from torch.utils.data import DataLoader

from data_utils import Vocabulary, TranslationDataset, create_synthetic_data

def collate_fn(batch):
    """Custom collate function for dynamic padding"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    tgt_output_batch = [item['tgt_output'] for item in batch]
    
    # Pad sequences to max length in batch
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    tgt_output_batch = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, batch_first=True, padding_value=0)
    
    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'tgt_output': tgt_output_batch
    }

def main():
    """Test data preparation pipeline"""    
    # TODO: Create synthetic data with 500 samples
    src_sentences, tgt_sentences = create_synthetic_data(num_samples=500)
    
    # TODO: Print the number of sentence pairs created
    print(f"Created {len(src_sentences)} sentence pairs")
    
    # TODO: Print sample data - show first 3 source and target pairs
    print("Sample data:")
    for i in range(3):
        print(f"  Source: {src_sentences[i]}")
        print(f". Target: {tgt_sentences[i]}")
    
    # TODO: Build vocabularies for source and target sentences
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    # TODO: Print vocabulary sizes
    print(f"Source vocabulary size: {src_vocab.size}")
    print(f"Target vocabulary size: {tgt_vocab.size}")
    
    # TODO: Create dataset and dataloader with batch size 4
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # TODO: Get the first batch from the dataloader
    batch = next(iter(dataloader))
    
    # TODO: Print the shapes of source, target, and target output tensors
    print(f"Batch source shape: {batch['src'].shape}")
    print(f"Batch target shape: {batch['tgt'].shape}")
    print(f"Batch target output shape: {batch['tgt_output'].shape}")
    
    # TODO: Decode the first sample in the batch and print the results
    sample_src = batch['src'][0]
    sample_tgt = batch['tgt_output'][0]
    
    decoded_src = src_vocab.decode(sample_src.tolist())
    decoded_tgt = tgt_vocab.decode(sample_tgt.tolist())
    
    print(f"Decoded source: '{decoded_src}'")
    print(f"Decoded target: '{decoded_tgt}'")

if __name__ == "__main__":
    main()