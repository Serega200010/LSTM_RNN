import sys


#Функцция предобработки текста
def preprocessed(text_to_process):
    lower_case_text = text_to_process.lower()
    processed_text = ""
    for symbol in lower_case_text:
        if "a" <= symbol <= "z" or "а" <= symbol <= "я" or "0" <= symbol <= "9" or symbol == " " or symbol == "\n" or symbol == '<' or symbol == '>':
            processed_text += symbol
        else:
            processed_text += " " + symbol + " "
    return processed_text.strip().replace(" ", " ").replace("\t", "")

#Функция, аналогичная zip, позволяющая принять в качестве аргумента список списков
def Zip(L):
    for i in range(len(L[0])):
        sub_list = [L[j][i] for j in range(len(L))]
        yield sub_list
#Функция, аналогичная Zip, позволяющая объединять элементы исходных списков по n штук
def super_zip(L,n:int):
    for i in range(0,len(L[0])//n,n):
        sub_list = [[L[j][i+k] for k in range(n)] for j in range(len(L))]
        yield sub_list

#Функция токенизации предобработанного текста
def tokenization(text):
    return [word for word in text.split()] + ['<eos>']

#Функция токенизации нескольких текстов
def texts_to_tokens(texts):
    tokenized = []
    for text in texts:
        tokenized = tokenized + tokenization(text)
    return tokenized

#Функция разбиения списка на batch_size равных частей, последняя из которых может содержать лишь остаток элементов, и не быть равной остальным
def split_list(L, batch_size):
   yield L[0*len(L)//batch_size : (0+1)*len(L)//batch_size]
   for i in range(1,batch_size):
       yield L[i*(len(L)//batch_size) : (i+1)*(len(L)//batch_size)]

#На случай, если удобнее пользоваться списком, а не генератором - аналог предыдущей функции, но возвращает список
def split(L,batch_size):
    batches = []
    for batch in split_list(L,batch_size):
        batches.append(batch)
    return batches

#Функция разбиения текстов на batch_size частей
def split_text(texts,batch_size):
    return split(texts_to_tokens(texts),batch_size)

#Функция генерации батчей размерности (batch_size x 1) по исходным текстам
def batch_generator(texts,batch_size):
    ans = []
    for batch in Zip(split_text(texts,batch_size)):
       ans.append(batch)
    return ans

#Функция слияния двумерных массивов по axis = 0 (Ex.: merge( [[1],[2],[3]]  ,  [[4],[5],[6]], 2 ) == [ [1,4] , [2,5] , [3,6] ])
def merge(L, n):
    ans = []
    for i in range(0,len(L)//n):
        ans.append([[L[i*n+k][j] for k in range(n)] for j in range(len(L[0]))])
    return ans
    
#Функция генерации батчей размерности (batch_size x num_of_steps) по исходным текстам
def form_the_batches(texts, batch_size, num_of_steps):
    return merge([b for b in batch_generator(texts,batch_size)],num_of_steps)


#Функция генерации батчей и целевых батчей по текстам из файла
def batch_gen(path : str,batch_size,num_steps):
    texts = []
    with open(path,'r') as fp:
        for text in fp:
            texts.append(preprocessed(text))
        bg = form_the_batches(texts, batch_size, num_steps)
        target_texts = texts
        target_texts[0] = target_texts[0].replace(target_texts[0].split()[0], '')
        target_texts[-1] = target_texts[-1] + ' <eos>'
        targets = form_the_batches(target_texts,batch_size, num_steps)
        return bg, targets


#Формирование словаря по текстам из файла
def make_vocab(path : str):
    texts = []
    vocab = []
    with open(path,'r') as fp:
        for text in fp:
            for word in preprocessed(text).split():
                vocab.append(word)
    vocab = set(vocab)
    return vocab

#формирование словаря вида {'word' : word_index_in_vocab} по словарю
def word2id(voc):
    w2i = {j : i for i,j in enumerate(voc)}
    return w2i


#формирование списка индексов слов в тексте по тексту
def text2id(text,w2i):
    t2i = []
    for word in text.split():
        try:
            t2i.append(w2i[word])
        except KeyError:
            t2i.append(w2i['<unk>'])
    return t2i


#формирование списка индексов слов в батче по батчу
def batch2id(Batch, w2i):
 b2i = []
 for batch in Batch:
    b2i.append([])
    for string in batch:
        b2i[-1].append([])
        for word in string:
            try:
                b2i[-1][-1].append(w2i[word])
            except KeyError:
                b2i[-1][-1].append(w2i['<unk>'])
 return b2i


#Объединение двух словарей
def merge_vocs(vocs):
    V = []
    for voc in vocs:
        V = V + voc
    V = set(V)
    return V

#Формирование батчей, содержищих не слова, а их индексы в словаре
def form_numeric_batches(path:str, batch_size, num_of_steps):
    A,B = batch_gen(path,batch_size,num_of_steps)
    voc = make_vocab(path)
    w2i = word2id(voc)
    num_A = batch2id(A,w2i)
    num_B = batch2id(B,w2i)
    return num_A,num_B, voc, w2i 

#Формирование полного набора батчей по текстам из файла
def full_extraction(path:str, batch_size, num_of_steps):
    A,B,voc,w2i = form_numeric_batches(path, batch_size, num_of_steps)
    num_batches = []
    for i in range(len(A)):
        num_batches.append( [A[i],B[i]])
    return num_batches,voc,w2i 

 
 
#def full_batch_gen(path : str,batch_size,num_steps)

#batches тут - результат выполнения batch_gen, то есть кортеж из двух генераторов, элементы которых - батчи со словами
'''
def batches2id(batches,w2i):
    B2I = []
    TARGETS = []
    print('B2I Start\n\n')
    print('#B2I__Batches: ', batches)
    print('#B2I__Batches[0]: ', batches[0])
    for batch in batches[0]:
        print('#B2I__batch in Batches[0]:',batch)
        b2i = batch2id(batch,w2i)
        B2I.append(b2i)
    for target_batch in batches[1]:
        t2i = batch2id(batch,w2i)
        TARGETS.append(t2i)
    return B2I, TARGETS

'''