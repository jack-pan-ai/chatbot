from corpus.chatbot_corpus import prepare_xiaohuangji
import config
from utils.word_sequence import Word_Sequence
import pickle

def ws_save():
    ws = Word_Sequence()
    with open(config.chat_input_path) as f:
        for line in f.readlines():
            ws.fit(line.strip().split())
        ws.build_vocab()
        print('The size of input vocabulary: ', len(ws))
        pickle.dump(ws, open(config.chatbot_ws_input_path, 'wb'))

    ws = Word_Sequence()
    with open(config.chat_target_path) as f:
        for line in f.readlines():
            ws.fit(line.strip().split())
        ws.build_vocab()
        print('The size of target vocabulary: ', len(ws))
        pickle.dump(ws, open(config.chatbot_ws_target_path, 'wb'))

if __name__ == '__main__':
    # by_word did not use the package of jieba, which is way faster than functions (lcut) in jieba
    num = prepare_xiaohuangji(by_word=True)
    print('Total number of the pairs in the corpus: ', num)
    ws_save()