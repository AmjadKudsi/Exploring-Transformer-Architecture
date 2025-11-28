import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

class Vocabulary:
    def __init__(self):
        # TODO: Initialize the token2idx dictionary with special tokens
        # Special tokens should be: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
        pass
    
    def add_token(self, token):
        # TODO: Add a new token to the vocabulary if it doesn't exist
        # Update both token2idx and idx2token mappings
        # Don't forget to update the size
        pass
    
    def build_vocab(self, sentences, min_freq=1):
        """Build vocabulary from list of sentences"""
        # TODO: Count word frequencies using Counter
        # TODO: Add tokens that appear at least min_freq times
        pass
    
    def encode(self, sentence):
        """Convert sentence to token indices"""
        # TODO: Split the sentence into tokens
        # TODO: Convert each token to its index, use <UNK> for unknown tokens
        pass
            
    def decode(self, indices):
        """Convert token indices back to sentence"""
        # TODO: Convert indices back to tokens, skip <PAD> tokens
        # TODO: Join tokens into a sentence and clean up special tokens
        pass

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Encode sentences
        src_indices = self.src_vocab.encode(src_sentence)
        tgt_indices = self.tgt_vocab.encode('<SOS> ' + tgt_sentence) 
        tgt_output = self.tgt_vocab.encode(tgt_sentence + ' <EOS>')
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long)
        }

def create_synthetic_data(num_samples=1000):
    """Create synthetic translation data (English to reversed English)"""
    templates = [
        "hello world", "good morning", "how are you", "nice to meet you",
        "thank you very much", "see you later", "have a nice day"
    ]
    
    src_sentences = []
    tgt_sentences = []
    
    for _ in range(num_samples):
        template = random.choice(templates)
        src_sentences.append(template)
        tgt_sentences.append(" ".join(template.split()[::-1]))
    
    return src_sentences, tgt_sentences