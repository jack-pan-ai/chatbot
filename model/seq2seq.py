import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, target, input_length, target_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        decoder_output, decoder_hidden = self.decoder(target, encoder_hidden)

        return decoder_output, decoder_hidden

    def evaluate(self, input, input_length):
        # encoder_oupt is not used
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        indices = self.decoder.evaluate(encoder_hidden)

        return indices

if __name__ == '__main__':
    seq2seq = Seq2Seq()
    print(seq2seq.parameters())