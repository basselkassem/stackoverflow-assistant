#%%
from numpy import vectorize
from io_handler import IOHandler
from config import config
from sklearn.metrics import pairwise_distances_argmin_min

class Ranker(object):
    def __init__(self):
        super().__init__()
        self.load_data()
    
    def load_data(self):
        data = IOHandler.deserialize(config['topic_tfidf_dataset'])
        self.post_ids = IOHandler.deserialize(config['post_id_ds'])
        self.question_vecs = data['X']
        self.tags = data['y']
    
    def find_best_post(self, question_vec, tag):
        idx = self.tags == tag
        post_tag_vecs = self.question_vecs[idx]
        post_tag_ids = self.post_ids[idx]
        best_post, _ = pairwise_distances_argmin_min(
            question_vec, post_tag_vecs, metric = 'cosine'
        )
        best_post = best_post[0]
        return post_tag_vecs[best_post], post_tag_ids[best_post]

# vectorizer = IOHandler.deserialize(config['topic_vec_path'])

# question =['how to declare a class']
# question_vec = vectorizer.transform(question)
# print(question_vec.shape)
# ranker = Ranker()
# match_vec, match_id = ranker.find_best_post(question_vec, 'python')
# match = vectorizer.inverse_transform(match_vec)[0]
# print(' '.join(match))
# print(match_id)
#%%
# %%
