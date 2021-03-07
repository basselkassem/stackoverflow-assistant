
#%%
import os
curr_dir = os.getcwd()
os.chdir('..')
curr_dir = os.getcwd()
resource_dir = os.path.join(curr_dir, 'resources')
data_dir = os.path.join(curr_dir, 'data')
model_dir = os.path.join(curr_dir, 'models')
config = {
    'posts_path': os.path.join(data_dir, 'tagged_posts.tsv'),
    'dialogues_path': os.path.join(data_dir, 'dialogues.tsv'),
    'intent_tfidf_dataset': os.path.join(data_dir, 'intent_tfidf_dataset.pkl'),

    'stop_words_path': os.path.join(resource_dir, 'stopwords.txt'),
    'intent_vec_path': os.path.join(resource_dir, 'intent_vectorizer.pkl'),

    'intent_lg_cls': os.path.join(model_dir, 'intent_lg_cls.pkl'),
}
# %%
