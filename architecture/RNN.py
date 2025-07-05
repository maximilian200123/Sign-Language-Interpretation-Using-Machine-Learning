import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, return_sequences=False, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.dropout_layer = nn.Dropout(dropout)
        
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, lengths):
        batch_size, max_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        
        for t in range(max_len):
            x_t = x[:, t, :]
            mask = (t < lengths).unsqueeze(1).float()
            gates = self.W_x(x_t) + self.W_h(h_t)
            i_t, f_t, o_t, g_t = gates.chunk(4, dim=1)
            
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(g_t)
            
            c_t_candidate = f_t * c_t + i_t * g_t
            h_t_candidate = o_t * torch.tanh(c_t_candidate)
            
            c_t = mask * c_t_candidate + (1 - mask) * c_t
            h_t = mask * h_t_candidate + (1 - mask) * h_t
           
            if self.return_sequences:
                outputs.append(h_t.unsqueeze(1))
       
        if self.return_sequences:
            output_seq = torch.cat(outputs, dim=1)
            output_seq = self.dropout_layer(output_seq)
            return output_seq, (h_t, c_t)
       
        else:
            h_t = self.dropout_layer(h_t)
            return h_t, c_t

class HandSignClassifier(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_classes=10, dropout=0.3):
        super().__init__()
        self.rnn1 = RNN(input_size, hidden_size, return_sequences=True, dropout=dropout)
        self.rnn2 = RNN(hidden_size, hidden_size, return_sequences=False, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        out_seq, _ = self.rnn1(x, lengths)
        h_t, _ = self.rnn2(out_seq, lengths)
        return self.classifier(h_t)
