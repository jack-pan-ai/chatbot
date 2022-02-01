import config
from tqdm import tqdm
from lib import cut
import string

def filter(pair):
    '''
    used to filter the pairs, that is, to filter questions or answers both of which are not proper.)
    :param pair: every pair of Q&A
    :return: good pairs
    '''
    if pair[0].lower() in list(string.ascii_lowercase):
        # if there is only one word in the pairs
        return True
    elif pair[1].count('=') > 2:
        return True


def prepare_xiaohuangji(by_word = False):
    'the path of raw data (raw text), with a lot of fucks'
    cur_path = config.cur_path
    path = cur_path + r'/corpus/orgin_corpus/xiaohuangji50w_nofenci.conv.txt'
    # print(path)
    if not by_word:
        path_input = cur_path + r'/corpus/processed_corpus/input.txt'
        path_target = cur_path + r'/corpus/processed_corpus/target.txt'
    else:
        path_input = cur_path + r'/corpus/processed_corpus/input_by_word.txt'
        path_target = cur_path + r'/corpus/processed_corpus/target_by_word.txt'

    f_input = open(path_input, 'a')
    f_target = open(path_target, 'a')

    one_pair = []
    num = 0
    with open(path, encoding='utf-8') as fr:
        for line in tqdm(fr.readlines(), desc='The process of processing the dataset (小黄鸡): '):
            if line.startswith('E'):
                continue
            else:
                line = line[1:].strip()
                # input the sentence without the front caption
                line = cut(line, by_word = by_word)
                line = ' '.join(line) + '\n'
                if len(one_pair) < 2:
                    one_pair.append(line)

                # here we want to process the pair with input and target together
                if len(one_pair) == 2:
                    if filter(one_pair):
                        one_pair = []
                        continue

                    # save the pair
                    num += 1
                    f_input.write(one_pair[0])
                    f_target.write(one_pair[1])
                    one_pair = []

        f_input.close()
        f_target.close()

    return num