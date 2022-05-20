from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import pickle
import numpy as np
from keras.models import model_from_json

marbert_model_path = 'UBC-NLP/MARBERT'
tokenizer = AutoTokenizer.from_pretrained(marbert_model_path, from_tf=True)
marbert_model = TFAutoModel.from_pretrained(
    marbert_model_path, output_hidden_states=True)

bert_model_path = 'bert-base-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path, from_tf=True)
bert_model = TFAutoModel.from_pretrained(bert_model_path, output_hidden_states=True)

off_scaler = pickle.load(open('./Models/offensive_scaler.pkl', 'rb'))
off_lr_model = pickle.load(open('./Models/lr_offensive_model.sav', 'rb'))

hs_scaler = pickle.load(open('./Models/hs_scaler.pkl', 'rb'))
hs_lr_model = pickle.load(open('./Models/lr_hs_model.sav', 'rb'))

with open('./Models/depression_model.json', 'r') as json_model:
    depression_model = model_from_json(json_model.read())
depression_model.load_weights('./Models/model_weights.h5')

def __bert_tokenize(text: str, tokenizer) -> list:
    max_len = len(tokenizer.tokenize(f'[CLS] {text} [SEP]'))
    tokens = tokenizer(text, padding='max_length',
                       truncation=True, max_length=max_len)
    return (np.expand_dims(np.array(tokens['input_ids']), 0), np.expand_dims(np.array(tokens['attention_mask']), 0), np.expand_dims(np.array(tokens['token_type_ids']), 0))

def __get_embeddings(tokens):
    ids = tf.convert_to_tensor(tokens[0])
    mask = tf.convert_to_tensor(tokens[1])
    type_ids = tf.convert_to_tensor(tokens[2])
    hidden_states = marbert_model(
        input_ids=ids, attention_mask=mask, token_type_ids=type_ids)[2]
    sentence_embd = tf.reduce_mean(tf.reduce_sum(
        tf.stack(hidden_states[-4:]), axis=0), axis=1)
    return sentence_embd

def __get_embeddings_depression(tokens):
    ids = tf.convert_to_tensor(tokens[0])
    mask = tf.convert_to_tensor(tokens[1])
    type_ids = tf.convert_to_tensor(tokens[2])
    pooled_output = bert_model(input_ids=ids, attention_mask=mask, token_type_ids=type_ids).last_hidden_state
    sentence_embd = tf.reduce_mean(pooled_output, axis=1)
    return sentence_embd

def __get_features(text, tokenizer):
    inputs = __bert_tokenize(text, tokenizer)
    return __get_embeddings(inputs)

def __get_features_depression(text, tokenizer):
    inputs = __bert_tokenize(text, tokenizer)
    return __get_embeddings_depression(inputs)

def make_offensive_prediction(text):
    features = __get_features(text, tokenizer)
    features = off_scaler.transform(features)
    return off_lr_model.predict(features)


def make_hs_prediction(text):
    features = __get_features(text, tokenizer)
    features = hs_scaler.transform(features)
    return hs_lr_model.predict(features)


def make_depression_prediction(text):
    inputs = __get_features_depression(text, bert_tokenizer)
    return np.argmax(depression_model.predict(inputs), axis=-1)
