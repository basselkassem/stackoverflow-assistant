#%%
from preprocess import Preprocessor
from io_handler import IOHandler
from config import config
import numpy as np

preprocessor = Preprocessor()

def create_intent_tfidf(size = 40000):
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
    

    intent_X, vectorizer = preprocessor.get_tf_idf(list(intent_txts), dump_vec = True)
    dataset = {'X': intent_X, 'y': intent_target}
    IOHandler.serialize(dataset, config['intent_tfidf_dataset'])

create_intent_tfidf(size = 40000)
#%%