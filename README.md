# KoreanTextGenerator

2020-2학기 정보검색(NLP) 수업에서 실시하였던 최종 프로젝트 코드입니다. Numpy(Cupy) 라이브러리 만을 활용하여 LSTM 모델을 구축해 보고 모델이 잘 구축되어있는지 확인하기 위해 모델을 이용하여 불완전한 문장이 입력되었을 때 그 다음에 나올 단어들을 예측하여 문장의 끝이 나올 때 까지 단어를 출력하는 Text Generation을 수행하였습니다.

## Dataset

#### 1. Data Source
국립국어원 모두의 말뭉치 사이트에 있는 국립국어원 신문 말뭉지(.ver 1)을 사용하였고, 그 중 첫 번째 파일을 사용함

#### 2. Data Format
데이터 파일은 JSON 파일로, 각 document에 문서를 식별할 수 있는 id, title, author와 같은 신문에 대한 정보, 신문의 내용인 paragraph로 이루어져 있으며, 이 중 내용이 담긴 paragraph에 있는 부분을 데이터로 사용하였다.

#### 3. Data Preprocessing
paragraph에 있는 문장의 내용중 기사의 제목, 특수문자, 문장의 끝을 나타내는 온점이 아닌 점들을 제거하였다. 이번 프로젝트에서는 온점을 end token처럼 사용하였기에 이러한 방식을 사용하였다.

#### 4. Data Tokenization and Padding
Keras에 있는 Tokenizer를 사용해 각 단어를 토큰화한다. 그리고 문장마다 모두 길이가 다르기 때문에 길이를 맞춰주기 위하여 제로패딩한다. 


## Reference
min_char_rnn : https://gist.github.com/karpathy/d4dee566867f8291f086#file-min-char-rnn-py