import spacy

#start
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')
doc = nlp("Tea is healthy and calming, don't you think?")

#tokenizing
for token in doc:
	print(token)

#text preprocessing
print(f"Token \t\t\tLemma \t\t\tStopword".format('Token', 'Lemma', 'StopWord'))
print('-'*40)
for token in doc:
	print(f"{str(token)}\t\t\t{token.lemma_}\t\t\t{token.is_stop}")

#patter matching
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

term = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
pattern = [nlp(text) for text in term]
matcher.add("TerminologyList", pattern)
text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.")
matches = matcher(text_doc)
print(matches)

match_id, start, end = matches[1]
print(nlp.vocab.strings[match_id], text_doc[start:end])