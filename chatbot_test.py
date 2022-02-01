from utils.datasets import train_data_loader

if __name__ == '__main__':
    for index, (input, target, input_len, target_len) in enumerate(train_data_loader):
        print(index, input, target, input_len, target_len)
        break