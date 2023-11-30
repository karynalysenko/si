class OneHotEncoder:
    def __init__(self, padder, max_length = None):
        self.padder = padder
        self.max_length = max_length
        self.alphabet = []
        self.char_to_index = {}
        self.index_to_char = {}
    
    def fit(self, data:list[str]):
        self.alphabet = sorted(set(char for string in data for char in string if char.isalpha()))
        self.alphabet.extend(self.padder)
        for i in range(len(self.alphabet)):
            self.index_to_char[i] = self.alphabet[i]
            self.char_to_index[self.alphabet[i]] = i

        if self.max_length is None:
            self.max_length = int(max(len(seq) for seq in data))
            # self.max_length = len(data)

    def transform(self, data:list[str]):
        encoded_seqs = []
        for seq in data[:self.max_length]:
            encoded_seq = [self.padder] * len(self.alphabet)
            for char in seq:
                if char in self.char_to_index:
                    encoded_seq[self.char_to_index[char]] = 1
            encoded_seqs.append(encoded_seq)
        return encoded_seqs


    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        decoded_sequences = []

        for encoded_seq in data:
            decoded_seq = [self.index_to_char[i] for i, val in enumerate(encoded_seq) if val == 1]
            decoded_sequences.append("".join(decoded_seq))

        return decoded_sequences
    
if __name__ == '__main__':
    # Instantiate the encoder
    encoder = OneHotEncoder(padder='_')

    # Fit the encoder to the data
    data = ['abc', 'def', 'ghax']

    encoded_data = encoder.fit_transform(data)
    print(encoded_data)
    # # Inverse transform to get back the original sequences
    decoded_data = encoder.inverse_transform(encoded_data)
    print(decoded_data)