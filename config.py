import pickle
import os
import torch

######################   Path  ######################
# cur_path = '/Users/panq/Documents/Deep_learning/test/chatbot'
cur_path = '.'
###################### Saving dict ####################
chatbot_by_word = True

# used for txt file for input and target
if chatbot_by_word:
    chat_input_path = cur_path + r'/corpus/processed_corpus/input_by_word.txt'
    chat_target_path = cur_path + r'/corpus/processed_corpus/target_by_word.txt'
    max_input_len = 30
    max_target_len = 30
    model_save_path = cur_path + r'/model/parameters/seq2seq_by_word.model'
    optimizer_save_path = cur_path + r'/model/parameters/optim_by_word.model'
else:
    chat_input_path = cur_path + r'/corpus/processed_corpus/input.txt'
    chat_target_path = cur_path + r'/corpus/processed_corpus/target.txt'
    max_input_len = 12
    max_target_len = 12
    model_save_path = cur_path + r'/model/parameters/seq2seq.model'
    optimizer_save_path = cur_path + r'/model/parameters/optim.model'

# used for rb file for input and output
if chatbot_by_word:
    chatbot_ws_input_path = cur_path + r'/corpus/binary_corpus/ws_input.pkl'
    chatbot_ws_target_path = cur_path + r'/corpus/binary_corpus/ws_target.pkl'

if os.path.exists(chatbot_ws_input_path):
    '''
    This is used to load dictionary and transform methods
    '''
    chatbox_ws_input = pickle.load(open(chatbot_ws_input_path, 'rb'))
    chatbox_ws_target = pickle.load(open(chatbot_ws_target_path, 'rb'))

batch_size = 128
embeddings_dim = 512
num_layers = 4
hidden_size = 256
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 10

teacher_forcing_ratio = 0.5