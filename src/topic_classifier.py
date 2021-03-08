#%%
from inspect import classify_class_attrs
from io_handler import IOHandler
from config import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from create_dataset import create_topic_tfidf_ds


class TopicClassifier(object):
    def __init__(self):
        super().__init__()
        self.model = OneVsRestClassifier(
            LogisticRegression(
                max_iter = 200,
            )
        )
        self.load_data()
        
    def load_data(self, ):
        data = IOHandler.deserialize(config['topic_tfidf_dataset'])
        X, y = data['X'], data['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size = 0.9, stratify = y,
        )

    def train(self,):
        self.model.fit(self.X_train, self.y_train)
        training_score = self.model.score(self.X_train, self.y_train)
        print(training_score)

    def test(self, ):
        y_test_hat = self.model.predict(self.X_test,)
        print(classification_report(
            self.y_test, y_test_hat, 
        ))
    def save(self,):
        IOHandler.serialize(
            self.model, config['topic_lg_cls']
        )

create_topic_tfidf_ds(size = 40000 * 4)
topic_cls = TopicClassifier()
topic_cls.train()
topic_cls.test()
topic_cls.save()

# %%
