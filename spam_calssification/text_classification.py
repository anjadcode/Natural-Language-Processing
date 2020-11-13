import pandas as pd
#load data
spam = pd.read_csv('spam.csv', encoding='latin-1')
print(spam.head(10))
print(spam.describe())

#make pipe for text categorizer
import spacy
#create empty model
spacy.prefer_gpu()
nlp = spacy.blank('en')

#create the text categorizer pipe, pipe itu class untuk memporses, menocndigurasi teoken
textcat = nlp.create_pipe(
	'textcat',
	config={"exclusive_classes" : True, "architecture" : 'bow' })

nlp.add_pipe(textcat)

textcat.add_label('ham')
textcat.add_label('spam')
train_text = spam['v2'].values
print(train_text)

train_label = [{'cats' :{'ham': label == 'ham', 'spam':label == 'spam'}} for label in spam['v1']]
print (train_label)

train_data = list(zip(train_text,train_label))
print (train_data[:3])

#create mini batch
from spacy.util import minibatch
import random

spacy.util.fix_random_seed()
optimizer = nlp.begin_training()

#create batch generator with mini batch
losses={}
for epoch in range(10):
	random.shuffle(train_data)
	batches = minibatch(train_data, size=8)
	for batch in batches:
		# Each batch is a list of (text, label) but we need to
	    # send separate lists for texts and labels to update().
	    # This is a quick way to split a list of tuples into lists
	    texts, labels = zip(*batch)
	    nlp.update(texts,labels, sgd=optimizer, losses=losses)
	print(losses)

text = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA",
         "anjiiiiiiiiiiiiiiiiiiiiiiiiiiiiing luuuuuu"]
doc = [nlp.tokenizer(t) for t in text]

#predict and score
textcat = nlp.get_pipe('textcat')
scores,_ = textcat.predict(doc)

print(scores)

#find label with highes score
predict_labels = scores.argmax(axis=1)
print([textcat.labels[int(label)] for label in predict_labels])