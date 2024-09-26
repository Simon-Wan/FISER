import os.path as osp


def get_tokenizers_and_vocabs(vocabulary_dir):
    src_tokenizer = Tokenizer(osp.join(vocabulary_dir, 'src_vocab.txt'))
    tgt_tokenizer = Tokenizer(osp.join(vocabulary_dir, 'tgt_vocab.txt'))
    src_vocab = src_tokenizer.vocab
    tgt_vocab = tgt_tokenizer.vocab

    q_tokenizer = Tokenizer(osp.join(vocabulary_dir, 'mid_q_vocab.txt'))
    s_tokenizer = Tokenizer(osp.join(vocabulary_dir, 'mid_s_vocab.txt'))
    v_tokenizer = Tokenizer(osp.join(vocabulary_dir, 'mid_v_vocab.txt'))
    o_tokenizer = Tokenizer(osp.join(vocabulary_dir, 'mid_o_vocab.txt'))
    mid_tokenizers = (q_tokenizer, s_tokenizer, v_tokenizer, o_tokenizer)
    mid_vocabs = {
        'q': q_tokenizer.vocab,
        's': s_tokenizer.vocab,
        'v': v_tokenizer.vocab,
        'o': o_tokenizer.vocab,
    }

    return src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, mid_tokenizers, mid_vocabs


class Tokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.vocab = dict()
        self.reverse_vocab = dict()
        self.load_vocab_dict()

    def get_len(self):
        return len(self.vocab)

    def load_vocab_dict(self):
        self.reverse_vocab[0] = ''
        with open(self.vocab_file, 'r') as f:
            vocab_list = f.readlines()
        for idx, token in enumerate(vocab_list):
            self.vocab[token.strip().lower()] = idx + 1
            self.reverse_vocab[idx + 1] = token.strip().lower()

    def single_tokenizer(self, token):
        token = token.lower()
        if token in self.vocab.keys():
            return self.vocab[token]
        else:
            print('NOT FOUND: {}'.format(token))
            return 0   # unknown token

    def sentence_tokenizer(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace(r'\n', ' ')
        sentence = sentence.replace('.', ' . ')
        sentence = sentence.replace(',', ' , ')
        sentence = sentence.replace('!', ' ! ')
        sentence = sentence.replace(';', ' ; ')
        sentence = sentence.replace(':', ' : ')
        sentence = sentence.replace('#', ' # ')
        sentence = sentence.replace('(', ' ( ')
        sentence = sentence.replace(')', ' ) ')
        sentence = sentence.replace('\'', ' \' ')
        sentence = sentence.replace('type-', 'type - ')

        ids = list()
        tokens = sentence.split()
        for token in tokens:
            if token in self.vocab.keys():
                ids.append(self.vocab[token])
            else:
                ids.append(0)   # unknown token
        return ids

    def translate(self, tensor, to_sentence=False):
        if len(tensor.shape) == 1:
            tokens = tensor.tolist()
            result = list()
            for tok in tokens:
                result.append(self.reverse_vocab[tok])
            if to_sentence:
                result = ' '.join(result)
            return result
        elif len(tensor.shape) == 2:
            batch_result = list()
            batch_tokens = tensor.tolist()
            for tokens in batch_tokens:
                result = list()
                for tok in tokens:
                    result.append(self.reverse_vocab[tok])
                if to_sentence:
                    result = ' '.join(result)
                batch_result.append(result)
            return batch_result
