import json
import re

import cupy as cp
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, SimpleRNN
from tensorflow.keras.models import load_model
from my_LSTM import softmax
from my_LSTM import my_LSTM_batch

test_word = ['남자', '그리고 그는', '대중교통 이용', '모든 순간',
             '새해 첫날', '텔레비전 광고',
             '크게', '대한민국 국민', '방송', '어린 아이']

def n_softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


def text_generation(model, word, npy):
    init_word = word
    cur_word_set = init_word
    result_stc = ''
    chk = 1
    word_cnt = 0
    perplexity = 1.0
    while 1:
        test_encoded = t.texts_to_sequences([cur_word_set])[0]
        test_sequences = pad_sequences([test_encoded], maxlen=max_len-1, padding='pre')
        result_p = model.predict(test_sequences)
        if npy:
            result_p = n_softmax(result_p)
            result = np.argmax(result_p)
            prob = result_p[0][result]
        else:
            result_p = softmax(result_p)
            result = cp.argmax(result_p)
            prob = result_p[result]
        perplexity = perplexity * (1/prob)
        for word, idx in t.word_index.items():
            if idx == result:
                word_cnt += 1
                if word == 'eos':
                    chk = max_len
                    break
                cur_word_set = cur_word_set + ' ' + word
                result_stc = result_stc + ' ' + word
                chk += 1
                break
        if chk is max_len:
            result_stc = result_stc + '.'
            print(init_word + result_stc)
            break
    print("Perplexity is %f" % perplexity ** (1 / word_cnt))
    return perplexity ** (1 / word_cnt)


def get_model_word():
    model1 = load_model(r'test3\RNN_Model_word.h5')
    model2 = load_model(r'test3\LSTM_Model_word.h5')
    p_lstm = 0
    p_rnn = 0
    for word in test_word:
        print("RNN RESULT -----------------")
        p_rnn += text_generation(model1, word, True)
        print("LSTM RESULT -----------------")
        p_lstm += text_generation(model2, word, True)
    print("SimpleRNN Test Perplexity Avg : %f" % (p_rnn/10))
    print("LSTM Test Perplexity Avg : %f" % (p_lstm/10))


def get_model_morphs():
    model1 = load_model(r'test1\RNN_Model_morphs.h5')
    model2 = load_model(r'test1\LSTM_Model_morphs.h5')
    p_lstm = 0
    p_rnn = 0
    for word in test_word:
        print("RNN RESULT -----------------")
        p_rnn += text_generation(model1, word, True)
        print("LSTM RESULT -----------------")
        p_lstm += text_generation(model2, word, True)
    print("SimpleRNN Test Perplexity Avg : %f" % (p_rnn/10))
    print("LSTM Test Perplexity Avg : %f" % (p_lstm/10))


def get_model_test():
    model1 = load_model(r'test2_Keras\RNN_Model.h5')
    model2 = load_model(r'test2_Keras\LSTM_Model.h5')
    model3 = my_LSTM_batch(128, 10, vocab_siz)
    model3.load_weight()
    p_lstm = 0
    p_rnn = 0
    p_my = 0
    for word in test_word:
        print("RNN RESULT -----------------")
        p_rnn += text_generation(model1, word, True)
        print("LSTM RESULT -----------------")
        p_lstm += text_generation(model2, word, True)
        print("MY_LSTM RESULT -----------------")
        p_my += text_generation(model3, word, False)
    print("SimpleRNN Test Perplexity Avg : %f" % (p_rnn / 10))
    print("LSTM Test Perplexity Avg : %f" % (p_lstm / 10))
    print("my_LSTM Test Perplexity Avg : %f" % (p_my / 10))


def model_train(e, b_size):
    model = Sequential()
    model.add(Embedding(vocab_siz, 10, input_length=max_len-1))
    model.add(SimpleRNN(128, input_shape=(1, max_len - 1)))
    model.add(Dense(vocab_siz, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model2 = Sequential()
    model2.add(Embedding(vocab_siz, 10, input_length=max_len-1))
    model2.add(LSTM(128, input_shape=(1, max_len - 1)))
    model2.add(Dense(vocab_siz, activation='softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    rnn_name = 'RNN_Model'
    lstm_name = 'LSTM_Model'

    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_siz)
    model.fit(X, y, epochs=e, batch_size=b_size)
    model2.fit(X, y, epochs=e, batch_size=b_size)

    model.save(rnn_name + '.h5')
    model2.save(lstm_name + '.h5')


def my_model_train(e, b_size):
    # 입력과 출력으로 데이터를 나눈다.
    X_t = sequences[:, :-1]
    y_t = sequences[:, -1]

    # 출력 데이터는 one-hot encoding 실시
    y_t = to_categorical(y_t, num_classes=vocab_siz)

    # hidden_layer, input_size(Embedding Size), output_size
    model = my_LSTM_batch(128, 10, vocab_siz)
    X_t = cp.asarray(X_t)
    y_t = cp.asarray(y_t)
    model.train_adam(X_t, y_t, e, b_size)
    model.save_weight()
    return model


def data_load(num, morph):

    # json 파일을 Load 한다.
    data = json.load(open("NIRW1900000001.json", 'r', encoding='UTF-8'))

    # 문장들을 list에 넣어주고, 정규표현식을 통해 정제를 해 준다.
    size = 0
    cnt = 0
    sentences = []
    for idx, i in enumerate(data['document']):
        if idx == num:
            break
        for idx2, j in enumerate(i['paragraph']):
            if idx2 is 0:
                continue
            text = re.sub('\(.*\)', '', j['form'])
            text = re.sub('\[.*\]', '', text)
            text = re.sub('\.\d', '', text)
            text = text.split(".")
            for k in text:
                if (len(k) > 5 and len(k) < 100):
                    k = re.sub('[/\",\`·\'<>;‘’→“”-]', '', k)
                    k += ' eos'
                    sentences.append(k)
                    cnt += 1
            size += len(j['form'])
    print("Number of Sentences is %d " % len(sentences))
    print("Number of Word is %d" % size)
    if morph:
        okt = Okt()
        for idx, stc in enumerate(sentences):
            sentences[idx] = okt.morphs(stc)

    return sentences


if __name__ == "__main__":
    cp.get_default_memory_pool().used_bytes()

    # 형태소 단위는 True, 단어 단위는 False, 앞의 숫자는 가져올 document 개수
    sentences = data_load(30, False)

    # 데이터 토큰화 과정
    t = Tokenizer()
    t.fit_on_texts(sentences)

    # 사전의 크기
    vocab_siz = len(t.word_index) + 1
    print("단어 사전의 크기는 %d" % vocab_siz)

    # 인덱스로 단어를 찾는 코드 생성
    index_to_word = {}
    for key, value in t.word_index.items():
        index_to_word[value] = key

    # 훈련 데이터셋을 만든다.
    sequences = list()
    for stc in sentences:
        encoded = t.texts_to_sequences([stc])[0]
        for i in range(1, len(encoded)):
            sequence = encoded[:i + 1]
            sequences.append(sequence)

    # 패딩을 위해 실제 길이를 구한다.
    max_len = max(len(l) for l in sequences)
    print(max_len)

    # 데이터 패딩을 통해 길이를 맞춰 준다.
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
    # 입력 데이터의 크기
    print("총 데이터의 개수는 %d" % len(sequences))
    get_model_test()

