class SimpleTokenizer:
    def __init__(self):
        self.token2id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id2token = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}

    def build_vocab(self, texts):
        idx = 4
        for text in texts:
            for word in text.strip().split():
                if word not in self.token2id:
                    self.token2id[word] = idx
                    self.id2token[idx] = word
                    idx += 1

    def encode(self, text):
        return [self.token2id.get(word, 1) for word in text.strip().split()]

    def decode(self, ids):
        return " ".join(self.id2token.get(i, "<UNK>") for i in ids)
texts = [
    "User: How many chairs in branch A?",
    "Bot: There are 10 chairs in branch A."
]

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(texts)

encoded = tokenizer.encode("User: How many chairs in branch A?")
print(encoded)
print(tokenizer.decode(encoded))
