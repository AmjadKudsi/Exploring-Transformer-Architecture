# implement the TranslationDataset class, which is responsible for loading and preparing sentence pairs for training
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

class Vocabulary:
    def __init__(self):
        self.token2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4
    
    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.size
            self.idx2token[self.size] = token
            self.size += 1
    
    def build_vocab(self, sentences, min_freq=1):
        """Build vocabulary from list of sentences"""
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence.split())
        
        for token, freq in counter.items():
            if freq >= min_freq:
                self.add_token(token)
    
    def encode(self, sentence):
        """Convert sentence to token indices"""
        tokens = sentence.split()
        return [self.token2idx.get(token, self.token2idx['<UNK>']) 
                for token in tokens]
            
    def decode(self, indices):
        """Convert token indices back to sentence"""
        tokens = [self.idx2token[idx] for idx in indices if idx != self.token2idx['<PAD>']]
        return ' '.join(tokens).replace('<SOS>', '').replace('<EOS>', '').strip()

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        # TODO: Store the sentences and vocabularies as instance variables
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        # TODO: Return the number of sentence pairs in the dataset
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # TODO: Get the source and target sentences at the given index
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # TODO: Encode the source sentence using src_vocab
        src_indices = self.src_vocab.encode(src_sentence)
        
        # TODO: Encode the target sentence with <SOS> token at the beginning using tgt_vocab
        tgt_indices = self.tgt_vocab.encode('<SOS> ' + tgt_sentence)
        
        # TODO: Encode the target sentence with <EOS> token at the end using tgt_vocab
        tgt_output = self.tgt_vocab.encode(tgt_sentence + ' <EOS>')
        
        # TODO: Return a dictionary with 'src', 'tgt', and 'tgt_output' keys
        # Each value should be a PyTorch tensor of type long
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