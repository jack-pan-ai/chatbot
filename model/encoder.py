import torch.nn as nn
import config
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.chatbox_ws_input),
                                      embedding_dim=config.embeddings_dim,
                                      padding_idx=config.chatbox_ws_input.PAD
                                      )
        self.gru = nn.GRU(
            input_size=config.embeddings_dim,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            batch_first=True
        )


    def forward(self, input, input_length):
        '''
        :param input: [batch_size, max_len]
        :return:
        '''
        embed = self.embedding(input) #[batch_size, max_len, embedding_dim]
        embed = pack_padded_sequence(embed, lengths= input_length, batch_first=True)
        out, hidden = self.gru(embed)
        out, out_length = pad_packed_sequence(out, batch_first=True, padding_value=config.chatbox_ws_input.PAD)
        # hidden: [num_layers, batch_size, hidden_size]
        # out: [batch_size, seq_len, hidden_size]
        return out, hidden
    
