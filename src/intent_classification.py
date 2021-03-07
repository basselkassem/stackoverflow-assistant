#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from io_handler import IOHandler
from sklearn.metrics import classification_report
from config import config
from create_dataset import create_intent_tfidf_ds

class IntentClassifier(object):
    def __init__(self, cls_name):
        super().__init__()
        self.load_data()
        self.model = LogisticRegression()
        
    def load_data(self, ):
        data = IOHandler.deserialize(config['intent_tfidf_dataset'])
        X, y = data['X'], data['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size = 0.9, stratify = y,
        )
  
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        training_score = self.model.score(self.X_train, self.y_train)
        print('Training Score:', training_score)

    def test(self):
        y_test_hat = self.model.predict(self.X_test)
        print( classification_report(
            self.y_test, y_test_hat,
        ))
    
    def save(self):
        IOHandler.serialize(
            self.model, config['intent_lg_cls']
        )

create_intent_tfidf_ds(size = 40000)
intent_cls = IntentClassifier('lg')
intent_cls.train()
intent_cls.test()
intent_cls.save()
# %%
