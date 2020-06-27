import numpy as np
import torch
import torch.nn as nn
import LSTM
from LSTM import LSTMLanguageModel 
import torch.nn.functional as F

def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def batch_to_gen(X,Y):
    for i in range(len(X)):
        x = torch.LongTensor(X[i]).view(1,-1)
        y = torch.LongTensor(Y[i]).view(1,-1)
        yield (x,y) 

#batch == (X,Y), X == батч примеров, Y == правильные слова, и то и другое - тензоры
#Функция, проводящая одну эпоху обучения модели, и возвращающая перплексию
def run_epoch(lr : float, model: LSTMLanguageModel, batches, w2i, loss_fn: nn.Module, optimizer: nn.Module = None, device: torch.device = None) -> float:
        total_loss, total_examples = 0.0, 0
        for batch in batches:
            X = torch.LongTensor(batch[0]).to(device)
            Y = torch.LongTensor(batch[1]).to(device)
            X = X.to(device)
            Y = Y.to(device)       
            if optimizer is not None:
                optimizer.zero_grad()
            initial_state = model.init_hidden(batch_size= len(batch[0]))
            initial_c = model.init_c(batch_size= len(batch[0]))
            for i in range(len(initial_state)):
                initial_state[i] = initial_state[i].to(device)
            for i in range(len(initial_c)):
                initial_c[i] = initial_c[i].to(device)
            logits= model(X, initial_state,initial_c)

            loss = loss_fn(logits.view((-1, model.vocab_size)), Y.view(-1))
            total_examples += loss.size(0)
            total_loss += loss.sum().item()
            loss = loss.mean()

            if optimizer is not None:
                update_lr(optimizer, lr)
                loss.backward()
                optimizer.step()
        return np.exp(total_loss / total_examples)

#Функция, проводящая обучение модели, и возвращающая list, содержащий в себе значения перплексий на каждой эпохе обучения
def train_model(learning_rates : list, model: LSTMLanguageModel, batches, w2i, device: torch.device = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    perp = []
    for lr in learning_rates:
        print('\nlr = ',lr)
        update_lr(optimizer,lr)
        epoch_perplexity = run_epoch(lr = lr,model = model,batches = batches, w2i = w2i, loss_fn = loss_fn, optimizer = optimizer, device = device)
        print(epoch_perplexity)
        perp.append(epoch_perplexity)
    return perp

#Функция, генерирующая тексты по обученной модели
def sample(model: LSTMLanguageModel, vocab: list, start_token: str,w2i: dict, temperature: float, max_len: int = 20,device: torch.device = None) -> str:
    text = []
    prev_id = w2i[start_token]
    model.eval()
    with torch.no_grad():   
        hidden_state = model.init_hidden(batch_size=1)#.to(device)
        for i in range(max_len):
            X = torch.tensor([[prev_id]], dtype=torch.int64, device=device)
            logits = model(X,model.init_hidden(1),model.init_c(1))
            softmax = F.softmax(logits, -1).cpu().numpy()[0, 0]
            if temperature != 1.0:
                softmax = np.float_power(softmax, 1.0 / temperature)
                softmax /= softmax.sum()
            prev_id = np.random.choice(list(range(len(softmax))), p=softmax)
            text.append(vocab[int(prev_id)])
            if vocab[prev_id] == '<eos>':
                break
    Text = ''
    for word in text:
        Text = Text+ ' ' + str(word)
    return Text



