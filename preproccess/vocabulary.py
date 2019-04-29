import json
import nltk

def get_vocaluary(data_entry_path, vocabulary_path):
    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)

    vocabulary = ['<S>', '</S>', '<EOS>']
    for key in data_dict.keys():
        sent_list = data_dict[key]
        for sent in sent_list:
            word_list = nltk.word_tokenize(sent)
            for word in word_list:
                if not vocabulary.__contains__(word):
                    vocabulary.append(word)
    print('vocabulary size = %s' % len(vocabulary))

    with open(vocabulary_path, 'w') as file:
        json.dump(vocabulary, file)

    print('vocabulary get success!')


vocabulary_path = '../data/vocabulary.json'
data_entry_path = '../data/data_entry.json'

get_vocaluary(data_entry_path, vocabulary_path)