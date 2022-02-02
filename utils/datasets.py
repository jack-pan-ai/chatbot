import config
import torch
from torch.utils.data import Dataset, DataLoader



class ChatbotDataset(Dataset):
    def __init__(self):

        self._input_path = config.chat_input_path
        self._target_path = config.chat_target_path
        self.input_lines = open(self._input_path).readlines()
        self.target_lines = open(self._target_path).readlines()

        assert len(self.input_lines) == len(self.target_lines), 'The length of input and target is not same.'

    def __getitem__(self, index):

        input = self.input_lines[index].strip().split()
        target = self.target_lines[index].strip().split()
        input_len = len(input) if len(input) < config.max_input_len else config.max_input_len
        target_len = len(target) if len(target) < config.max_target_len else config.max_target_len

        return input, target, input_len, target_len

    def __len__(self):
        return len(self.input_lines)

def collate_fn(batch):
    '''
    batch: [(input, target, input_len, target_len)] where input and output is still a text
    used to transform texts into numbers
    '''
    batch = sorted(batch, key=lambda x:x[-2], reverse=True)
    input, target, input_len, target_len = zip(*batch)
    input = [config.chatbox_ws_input.transform(i, max_len = config.max_input_len) for i in input]
    input = torch.LongTensor(input)
    target = [config.chatbox_ws_target.transform(i, max_len = config.max_target_len) for i in target]
    target = torch.LongTensor(target)
    input_len = torch.LongTensor(input_len)
    target_len = torch.LongTensor(target_len)

    return input, target, input_len, target_len


train_data_loader = DataLoader(ChatbotDataset(), batch_size= config.batch_size, shuffle=True, collate_fn=collate_fn)