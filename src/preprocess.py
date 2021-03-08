#%%
from io_handler import IOHandler
from config import config
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor(object):
    def __init__(self):
        super().__init__()
        self.load_resources()

    def load_resources(self):
        stop_words =  IOHandler.read_txt(config['stop_words_path'])
        self.stop_words = [x.strip().lower() for x in stop_words]

    def preprocess_txt(self, text, remove_stopwords = True):
        replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
        keep_symbols_re = re.compile('[^0-9a-z #+_]')
        res = text.lower()
        res = replace_by_space_re.sub(' ', res)
        res = keep_symbols_re.sub('', res)

        if remove_stopwords:
            res = self.remove_stop_words(res)

        return res.strip()
    
    def remove_stop_words(self, text):
        res = ' '.join([x for x in text.split() if x not in self.stop_words])
        return res.strip()
    
    def remove_duplicates(self, txt_df, cols = ['title', 'tag']):
        return txt_df.drop_duplicates(subset = cols, keep = 'first')
    
    def preprocess_txts(self, texts, remove_stopwords = True):
       return [self.preprocess_txt(txt) for txt in texts]
    
    def get_tf_idf(self, texts, vectorizer_path):
        tf_idf_vec = TfidfVectorizer(
            texts,
            ngram_range = (1, 2),
            max_df = 1.0,
            min_df = 0.0,
            #max_features = 32*100,
        )
        res = tf_idf_vec.fit_transform(texts)
        IOHandler.serialize(tf_idf_vec, vectorizer_path)
        return res

# %%
