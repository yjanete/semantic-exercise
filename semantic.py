import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1)) 


# the words 'cat' and 'monkey' are the ones with more similarity - maybe because both are animals?
# and the ones with less similarity are 'cat' and 'banana' - I know for a fact that cats dislike banana (my cat hates it!!)

# another example
word4 = nlp("fox")
word5 = nlp("cherry")
word6 = nlp("wolf")
print(word4.similarity(word5))
print(word6.similarity(word5))
print(word6.similarity(word4))

# the words 'fox' and 'wolf' are the ones with more similarity - They belong to the Canidae family
# and the ones with less similarity are 'wolf' and 'cherry'. But 'Fox' and 'cherry' do not have much similarity either. 
### QUESTION: In some scenarios (with different words) I obtained negative values for similarity. Is that correct?

## When running the example file with the simpler language model ‘en_core_web_sm’, the similarity found between elements is much higher.
# This is probably because this model's similarity method is based on the tagger, parser and NER, which may not give useful similarity judgements.
# 'en_core_web_sm' has no word vectors loaded and only yse context-sensitive tensors.
# The en_core_web_md is a larger model and this means it will usually take a little longer to load the model. 
## On the other hand, we usually expect it to find more entities.
# Most differences are obviously statistical. In general, we do expect larger models to be "better" and more accurate overall. 



# *******************************************************

# compare series of words with one another
tokens = nlp('cat apple monkey banana')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# ascertaining similarity between longer sentences
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)