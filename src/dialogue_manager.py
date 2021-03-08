#%%
from config import config
from io_handler import IOHandler
from preprocess import Preprocessor
from ranker import Ranker

class DialogueManager(object):
    def __init__(self,):
        self.preprocessor = Preprocessor()
        self.intent_vectorizer = IOHandler.deserialize(config['intent_vec_path'])
        self.topic_vectorizer = IOHandler.deserialize(config['topic_vec_path'])
        self.intent_cls = IOHandler.deserialize(config['intent_lg_cls'])
        self.topic_cls = IOHandler.deserialize(config['topic_lg_cls'])
        self.ranker = Ranker()

    def prepare_input(self, question):
        txt = self.preprocessor.preprocess_txt(question)
        self.intent_input = self.intent_vectorizer.transform([txt])
        self.topic_input = self.topic_vectorizer.transform([txt])

    def generate_answer(self, question):
        self.prepare_input(question)
        question_intent = self.intent_cls.predict(
            self.intent_input
        )
        if question_intent == 0:
            return 'No answer available'
        else:
            question_topic = self.topic_cls.predict(
                self.topic_input
            )
            matched_question, question_id = self.ranker.find_best_post(
                self.topic_input, question_topic
            )
            matched_question = ' '.join(
                self.topic_vectorizer.inverse_transform(matched_question)[0]
            )
            
            return question_topic[0], question_id, matched_question

dialogue_manager = DialogueManager()
questions = [
    'What is your name?',
    'how to create class in Java?',
    'system.out.println()',
    'how define a function in Python?',
    'How are you doing?',
    'how animate a div in a website?',
    'variable is not defined',
    'The weather is good',
]

for question in questions:
    answer = dialogue_manager.generate_answer(question)
    print(question, '-->>', answer)
    print()
#%%

# %%
