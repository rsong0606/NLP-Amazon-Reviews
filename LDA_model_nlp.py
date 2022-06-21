import data_pre_funs
import nltk
import emoji
import gensim
import spacy
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
import string
from pprint import pprint

df = pd.read_csv('dataset/AllProductReviews.csv')
print(df.head())
print(df.columns)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

df['ReviewTitle'] = df['ReviewTitle'].apply(data_pre_funs.remove_emoji_)
df['ReviewBody'] = df['ReviewBody'].apply(data_pre_funs.remove_emoji_)
df['ReviewBody'] = df['ReviewBody'].apply(data_pre_funs.remove_newline)
df['ReviewTitle'] = df['ReviewTitle'].apply(data_pre_funs.remove_newline)
df = df[df.ReviewTitle.apply(lambda x: len(str(x).split()) >= 3)]
df['ReviewBody'] = df['ReviewBody'].apply(data_pre_funs.remove_special_chars)
nltk.download('stopwords')
stop = stopwords.words('english')
df['ReviewBody'] = df['ReviewBody'].apply(
    lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
df = df.reset_index(drop=True)
print(df.tail())

texts = df.ReviewBody.values.tolist()


def process_words(texts):
    result = []
    for t in texts:
        t = ' '.join(re.findall(r'\b\w[\w\']+\b', t))
        doc = nlp(t)
        result.append([token.lemma_.lower() for token in doc])
    return result


processed_text = process_words(texts)

print(processed_text[:1])
dictionary = gensim.corpora.Dictionary(processed_text)
print(f'Number of unique tokens: {len(dictionary)}')

corpus = [dictionary.doc2bow(t) for t in processed_text]

num_topics = 5

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=num_topics,
                                            random_state=117, update_every=1,
                                            chunksize=1500,
                                            passes=5, iterations=10,
                                            alpha='asymmetric', eta=1 / 100,
                                            per_word_topics=True)

print(pprint(lda_model.print_topics(num_words=10)))


def get_main_topic_df(model, bow, texts):
    topic_list = []
    percent_list = []
    keyword_list = []

    for wc in bow:
        topic, percent = sorted(model.get_document_topics(wc), key=lambda x: x[1], reverse=True)[0]
        topic_list.append(topic)
        percent_list.append(round(percent, 3))
        keyword_list.append(' '.join(sorted([x[0] for x in model.show_topic(topic)])))

    result_df = pd.concat([pd.Series(topic_list, name='Dominant_topic'),
                           pd.Series(percent_list, name='Percent'),
                           pd.Series(texts, name='Processed_text'),
                           pd.Series(keyword_list, name='Keywords')], axis=1)

    return result_df


main_topic_df = get_main_topic_df(lda_model, corpus, processed_text)

df_out = pd.concat([df, main_topic_df], axis=1)
df_out.to_csv('dataset/lda_df.csv', index=False)
print(main_topic_df.tail())

# grouped_topics = main_topic_df.groupby('Dominant_topic')
# grouped_topics.count()['Processed_text']. \
#     plot.bar(rot=0). \
#     set(title=f'Dominant Topic Frequency in the {len(reviews)} Reviews',
#         ylabel='Topic frequency');
