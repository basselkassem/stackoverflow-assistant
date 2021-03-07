#%%
from preprocess import Preprocessor
from io_handler import IOHandler
from config import config
import numpy as np

preprocessor = Preprocessor()

def create_intent_tfidf_ds(size = 40000):
    stackoverflow_posts = IOHandler.read_txt_as_df(config['posts_path'])
    dialouge_posts = IOHandler.read_txt_as_df(config['dialogues_path'])

    stackoverflow_posts = preprocessor.remove_duplicates(stackoverflow_posts, cols = ['title'])
    dialouge_posts = preprocessor.remove_duplicates(dialouge_posts, cols = ['text'])

    stackoverflow_posts = stackoverflow_posts.sample(size)
    dialouge_posts = dialouge_posts.sample(size)

    dialouge_txts = dialouge_posts['text'].values
    dialouge_txts = preprocessor.preprocess_txts(dialouge_txts)
    dialouge_intent_target = np.zeros((len(dialouge_txts), ))

    stackoverflow_txts = stackoverflow_posts['title'].values
    stackoverflow_txts = np.random.choice(stackoverflow_txts, len(dialouge_txts))
    stackoverflow_txts = preprocessor.preprocess_txts(stackoverflow_txts)
    stackoverflow_intent_target = np.ones((len(stackoverflow_txts), ))

    intent_txts = np.concatenate([stackoverflow_txts, dialouge_txts])
    intent_target = np.concatenate([stackoverflow_intent_target, dialouge_intent_target])

    index = np.arange(0, len(dialouge_txts) * 2)
    np.random.shuffle(index)

    intent_txts = intent_txts[index]
    intent_target = intent_target[index]
    

    intent_X = preprocessor.get_tf_idf(list(intent_txts), config['intent_vec_path'])
    dataset = {'X': intent_X, 'y': intent_target}
    IOHandler.serialize(dataset, config['intent_tfidf_dataset'])

def create_topic_tfidf_ds(size = 40000):
    stackoverflow_posts = IOHandler.read_txt_as_df(config['posts_path'])
    stackoverflow_posts = preprocessor.remove_duplicates(stackoverflow_posts, cols = ['title'])

    stackoverflow_posts = stackoverflow_posts.sample(size)

    IOHandler.serialize(stackoverflow_posts['post_id'].values, config['post_id_ds'])
    stackoverflow_txts = stackoverflow_posts['title'].values
    stackoverflow_txts = preprocessor.preprocess_txts(stackoverflow_txts)
    y = stackoverflow_posts['tag'].values

    X = preprocessor.get_tf_idf(stackoverflow_txts, config['topic_vec_path'])
    IOHandler.serialize({'X': X, 'y': y }, config['topic_tfidf_dataset'])

#%%