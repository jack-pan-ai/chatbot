from utils.datasets import train_data_loader
from model.encoder import Encoder
from model.train import train

def trainloader_test():
    for index, (input, target, input_len, target_len) in enumerate(train_data_loader):
        print(index, input.shape, target.shape, input_len, target_len)
        break

def encoder_test():
    encoder = Encoder()
    for input, target, input_length, target_length in train_data_loader:
        out, hidden = encoder(input, input_length)
        print(hidden.shape)
        print(out.shape)
        break

def train_test():
    train()

if __name__ == '__main__':
    train_test()