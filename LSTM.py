import numpy as np
import torch
import torch.nn as nn
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu' 

#Создание класса LSTM - ячейки, унаследованного от nn.Module
#В конструкторе:
#               input_size - размерность векторов входных данных
#               hidden_size - размерность матриц скрытых состояний
class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMcell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_input = nn.Parameter(torch.Tensor(4*hidden_size,input_size))
        self.B_input = nn.Parameter(torch.Tensor(4*hidden_size))

        self.W_hidden = nn.Parameter(torch.Tensor(4*hidden_size,hidden_size))
        self.B_hidden = nn.Parameter(torch.Tensor(4*hidden_size))

        self.reset_parameters()

    def forward(self, inp: torch.Tensor, state: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        '''Описание размерностей введенных матриц:
        self.W_input: 4hidden_size x input_size (input size == embedding_size)
        self.W_hidden: 4hidden_size x hidden_size

        self.B_input, self.B_hidden:  4hidden_size x 1 => B_i, B_h: 4hidden_size x batch_size 
        inp: embedding_size x batch_size
        state:  hidden_size x batch_size
        '''
        
        B_i = torch.Tensor([list(self.B_input) for i in range(inp.shape[0])]).transpose(0,-1).to(device)#!!!
        B_h = torch.Tensor([list(self.B_hidden) for i in range(inp.shape[0])]).transpose(0,-1).to(device)#!!!
        tmp_matrix = torch.matmul(self.W_input,inp.transpose(0,-1))
        tmp_matrix = tmp_matrix + B_i
        tmp_matrix = tmp_matrix  + torch.matmul(self.W_hidden,state)
        tmp_matrix = tmp_matrix + B_h



        chunked = torch.chunk(tmp_matrix,4)
        i = torch.sigmoid(chunked[0])
        f = torch.sigmoid(chunked[1])
        g = torch.tanh(chunked[2])
        o = torch.sigmoid(chunked[3])

        c_1 = f*c + i*g
        h_1 = o*torch.tanh(c_1)

        return c_1,h_1

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


#Создание класса LSTM - слоя, содержащего реализованную ранее LSTM-ячейку
#В конструкторе:
#               emb_size: раземрность эмбеддингов
#               hidden_size: размерность вектора состояния LSTM-ячейки
#               vocab_size: мощность словаря
class LSTMLayer(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, vocab_size: int):
        super(LSTMLayer,self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.lstmcell = LSTMcell(input_size = emb_size, hidden_size = hidden_size)  
        self.decoder = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        self.init_weights()
    def lstmlayer(self, embeddings: torch.Tensor, state: torch.Tensor, c: torch.Tensor):
        outputs = []
        for timestep in range(embeddings.shape[0]):
            c, state = self.lstmcell(embeddings[timestep], state, c)
            outputs.append(state.transpose(0,-1))
        return torch.stack(outputs), state , c
    
    def forward(self, model_input: torch.Tensor, initial_state: torch.Tensor, initial_c: torch.Tensor, mode: str):
        if mode == 'start':
            embs = self.embedding(model_input).transpose(0, 1).contiguous()
        else:
            embs = model_input
        outputs, hidden, c = self.lstmlayer(embs, initial_state, initial_c)

        if mode == 'inter' or mode == 'start':
            return outputs, hidden, c
        
        if mode == 'final':
            logits = self.decoder(outputs).transpose(0, 1).contiguous()
            return logits, hidden, c
        else: 
            logits = self.decoder(outputs).transpose(0, 1).contiguous()
            return logits,outputs, hidden, c
    
    def init_weights(self):
        '''Weights initialization'''
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.decoder.weight.data.uniform_(-0.1, 0.1)



#Создание класса языковой модели, представляющей собой объединение нескольких LSTM-слоев
#В конструкторе:
#               emb_size: раземрность эмбеддингов
#               hidden_size: размерность вектора состояния LSTM-ячейки
#               vocab_size: мощность словаря
class LSTMLanguageModel(nn.Module):
    def __init__(self, emb_size : int, hidden_sizes: int , vocab_size : int ):
        super(LSTMLanguageModel, self).__init__()
        self.emb_size = emb_size
        self.hidden_sizes = hidden_sizes
        self.vocab_size = vocab_size
        self.num_layers = len(hidden_sizes)

        emb_sizes = [emb_size] + hidden_sizes
        self.layers = nn.ModuleList([LSTMLayer(emb_sizes[i], hidden_sizes[i],self.vocab_size )   for i in range(self.num_layers)])
    def init_hidden(self, batch_size: int):
        h_s = []
        for hidden_size in self.hidden_sizes:
            h_s.append(torch.zeros(batch_size, hidden_size).transpose(0,-1))
        return h_s
    
    def init_c(self, batch_size: int):
        c_s = []
        for hidden_size in self.hidden_sizes:
            c_s.append(torch.zeros(batch_size, hidden_size).transpose(0,-1))
        return c_s



    def forward(self,model_input,h_s,i_c):
        b_s = model_input.shape[0]
        outputs = []
        logits = []
        i = 1
        for layer in self.layers:
            if i == 1:
                outputs, hidden, c = layer(model_input,h_s[i-1], i_c[i-1],'start') #or layer.init_hidden(b_s)
            if i>1 and i<len(self.layers):
                outputs, hidden, c = layer(outputs,h_s[i-1], i_c[i-1],'inter')
            if i == len(self.layers):
                logits, hidden, c = layer(outputs,h_s[i-1], i_c[i-1],'final')
            i+=1
        return logits
