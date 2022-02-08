import torch.nn as nn
import torch
import config
import torch.nn.functional as F
from numpy import random

import matplotlib.pyplot as plt

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # len(config.chatbox_ws_target): there is only one dictionary which belongs to target dataset;
        # later we cannot incorporate the input vocabulary into target
        self.embedding = nn.Embedding(
            num_embeddings=len(config.chatbox_ws_target),
            embedding_dim=config.embeddings_dim,
            padding_idx=config.chatbox_ws_target.PAD,
        )
        self.gru = nn.GRU(
            input_size=config.embeddings_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, len(config.chatbox_ws_target))

    def forward(self, target, encoder_hidden):
        # 1. get initial hidden_state
        decoder_hidden = encoder_hidden
        # 2. get initial SOS
        batch_size = target.size(0)
        decoder_input = torch.ones([batch_size, 1], dtype=torch.int64) * config.chatbox_ws_target.SOS
        decoder_input = decoder_input.to(config.device)

        # save the whole prediction, where it's decoder output [batch_size, max_len, vocab]instead of seq[batch_size, len_seq]
        decoder_outputs = torch.zeros([batch_size, config.max_target_len, len(config.chatbox_ws_target)]).to(config.device)
        # decoder_outputs = torch.zeros([batch_size, len(config.num_sequence), config.max_len+1])

        # 3. establish a for-loop to predict the sentence
        if config.teacher_forcing_ratio < random.random():
            for t in range(config.max_target_len):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = decoder_output_t
                # teacher forcing in for-loop or out of for-loop
                decoder_input = target[:, t].unsqueeze(-1)
        else:
            for t in range(config.max_target_len):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = decoder_output_t
                # teacher forcing in for-loop or out of for-loop
                value, index = torch.topk(decoder_output_t, 1)
                decoder_input = index

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        '''
        need to do a prediction
        :param decoder_input: [batch_size, max_len]
        :param decoder_hidden: [1, batch_size, hidden_layer]
        :return: decoder_output [batch_size, vocab] and decoder_hidden for this time
        '''
        decoder_input_embeded = self.embedding(decoder_input)  # out: batch_size, 1, embeddings_dim
        # out: [batch_size, 1, hidden_size]
        # decoder_hidden = [1, batch_size, hidden_size]
        out, decoder_hidden = self.gru(decoder_input_embeded, decoder_hidden)

        out = out.squeeze(1)
        output = F.log_softmax(self.fc(out), dim=-1)  # [batch_size, vocab_size]
        return output, decoder_hidden

        # evaluation

    def evaluate(self, encoder_hidden):
        decoder_hidden = encoder_hidden
        batch_size = decoder_hidden.size(1)
        decoder_input = (torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64)) * config.chatbox_ws_target.SOS).to(
            config.device)

        indices = []
        for i in range(config.max_len + 7):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            values, index = torch.topk(decoder_output_t, 1)  # index: [batch_size, 1]
            decoder_input = index

            index = index.squeeze(1).cpu().detach().numpy()  # [batch_size, ]
            indices.append(index)  # adding by row

        return indices  # [ max_len+7, batch_size]

