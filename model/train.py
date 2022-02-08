import torch
from utils.datasets import train_data_loader
from model.seq2seq import Seq2Seq
from torch.optim import Adam
import config
import torch.nn.functional as F
from tqdm import tqdm

# 训练过程
# 1. instance the model, optimizer, loss
seq2seq = Seq2Seq()
seq2seq = seq2seq.to(config.device)
optimizer = Adam(seq2seq.parameters(), lr = 0.001)

# 2. iterate dataloader
# 3. output and true label -> loss
def train():
    for epoch in range(config.epochs):
        bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ascii=True, desc='train')
        for index, (input, target, input_len, target_len) in bar:
            input = input.to(config.device)
            target = target.to(config.device)
            # input_len = input_len.to(config.device)
            # target_len = target_len.to(config.device)
    
            decoder_output, _ = seq2seq(input, target, input_len, target_len)

            decoder_output = decoder_output.transpose(1,2)
            loss = F.nll_loss(decoder_output, target, ignore_index=config.chatbox_ws_target.PAD)
            # loss = F.nll_loss(torch.transpose(decoder_output, 2, 1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_description('epoch:{}\t index:{}\t loss:{:.3f}\t'.format(epoch, index, loss.item()))
    
            # 4. model save for reloading again
            if index%100 ==0:
                torch.save(seq2seq.state_dict(), config.model_save_path)
                torch.save(optimizer.state_dict(), config.optimizer_save_path)