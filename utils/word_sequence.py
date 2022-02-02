'''
establish a dictionary and method of transform sequence into number, and number into sequence
'''


class Word_Sequence:
    PAD_TAG = 'PAD'
    PAD = 0
    UNK_TAG = 'UNK'
    UNK = 1
    SOS_TAG = 'SOS'
    SOS = 2
    EOS_TAG = 'EOS'
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG : self.PAD,
                     self.UNK_TAG : self.UNK,
                     self.SOS_TAG : self.SOS,
                     self.EOS_TAG : self.EOS
                     }
        self.count = {}

    def __len__(self):
        return len(self.dict)

    def fit(self, sentence):
        '''
        statistical frequency of words
        '''
        for word in sentence:
            self.count[word] = self.count.get(word, 0)+ 1

    def build_vocab(self, min_count = 5, max_count = None, max_features = None):
        '''
        Build vocabulary
        '''
        temp = self.count.copy()
        # preprocessing
        for key in temp:
            cur_count = self.count.get(key, 0)
            if min_count is not None:
                if min_count > cur_count:
                    del self.count[key]
            if max_count is not None:
                if max_count < cur_count:
                    del self.count[key]
        if max_features is not None:
            self.count = dict(sorted( self.count.items() , key=lambda x:x[1], reverse=True)[:max_features])

        # dict for sequence to num
        for key in self.count:
            self.dict[key] = len(self.dict)
        # dict for num to sequence
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len = None, add_eos = True):
        '''Transform the sentence to numbers '''
        'Here is where we should go when facing the non-fixed sentence'
        # to fill the last as eos
        max_len = max_len - 1

        if len(sentence) > max_len:
            sentence = sentence[:max_len]

        sentence_len = len(sentence) # if true, then the len would be max_len, otherwise len(sentence)

        if add_eos:
            sentence = sentence + [self.EOS_TAG] # max_len+1 or len(sentence) +1

        if sentence_len < max_len:
            sentence = sentence + [self.PAD_TAG] * (max_len - sentence_len) # (max_len + 1 - (len(sentence) + 1))
        result = [self.dict.get(i, self.UNK) for i in sentence]

        return result

    def inverse_transform(self, indices):
        '''transform the numbers into sequence'''

        # result = [self.inverse_dict.get(i, self.PAD_TAG) for i in indices]
        result =[]
        for i in indices:
            if i == self.EOS:
                break
            result.append(self.inverse_dict.get(i, self.UNK_TAG))
        return result

if __name__ == '__main__':
    pass
