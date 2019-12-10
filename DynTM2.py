
import os, pickle
import numpy as np
import pandas as pd
import swifter

from bs4 import BeautifulSoup
from itertools import chain

import re, string, contractions, unicodedata

#Spacy
# import spacy
# spacy.load('en')
# from spacy.lang.en import English
# parser = English()

import nltk
nltk.data.path.append("/app/localstorage/u_am_coe/notebooks/nltk_data")
from nltk.corpus import stopwords
from nltk import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer 
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet


# Gensim
import gensim
from gensim.models import Phrases
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger

# Plots
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline



################## Transcript Parsing ##################

class Transcript:
    def __init__(self, title=None, date=None, companies=None):
        self.title = title
        self.date = date
        self.companies = companies
       
    def parse(self, df):
        soup = BeautifulSoup(df['content'], features="html.parser")
        self.title = soup.find('title').text
        self.date = soup.find('date').text
        self.companies = list(set([i.text for i in soup.find_all('company')]))
        self.p_map = {p.attrs['id']: p.text for p in soup.find_all('participant')}
        self.p_map['unknown'] = 'unknown'
        self.section_list = []
        for section in soup.find_all('section'):
            for speaker in section.find_all('speaker'):
                self.section_list.append({'speaker': self.p_map[speaker.attrs.get('id', 'unknown')],
                                     'section_type': section.attrs['name'],
                                     'type': speaker.attrs.get('type', 'N/A'),
                                     'text': [i.text for i in speaker.find_all('p')]})  
    def __str__(self):
        return "title={}, date={}".format(self.title, self.date)
    
    
    

    
################## Dynamic Topic Modelling ##################
    
class DynTM:   
    def __init__(self, stop_words=None, num_topics=None, company=None, 
                 struct_df=None, subset_df=None, uniquetimes=None, time_slices=None, time_df = None,
                 dictionary=None, corpus=None, texts=None, ldamodel=None):
        self.stop_words = stop_words
        self.struct_df = struct_df
        self.subset_df = subset_df
        self.uniquetimes = uniquetimes
        self.time_slices = time_slices
        self.time_df = time_df
#         self.vis_obj = vis_obj
        self.num_topics = num_topics
        self.company = company
        self.ldamodel = ldamodel
        self.dictionary = dictionary
        self.corpus = corpus
        self.texts = texts
        
    
    ################## Transcript Text Cleaning ##################
        
    def flatten(self, listOfLists):
        "Flatten one level of nesting"
        return list(chain.from_iterable(listOfLists))
    
    # Lemmatize with POS Tag
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        return stripped_text

    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def expand_contractions(self, text):
        return contractions.fix(text)

    def remove_special_characters(self, text, remove_digits=False):
        pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
        text = re.sub(pattern, '', text)
        return text

    def pre_process_document(self, document):        
        # converting to text
        document =  str(document)

        # strip HTML
        document = self.strip_html_tags(document)

        # lower case
        document = document.lower()

        # remove extra newlines (often might be present in really noisy text)
        document = document.translate(document.maketrans("\n\t\r", "   "))

        # remove accented characters
        document = self.remove_accented_chars(document)

        # remove special characters and\or digits    
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        document = special_char_pattern.sub(" \\1 ", document)
        document = self.remove_special_characters(document, remove_digits=True)  

        # remove extra whitespace
        document = re.sub(' +', ' ', document)
        document = document.strip()

        # Split the documents into tokens.
        tokenizer = RegexpTokenizer(r'\w+')
        document = document.lower()  # Convert to lowercase.
        document = tokenizer.tokenize(document)  # Split into words.

        # Remove numbers, but not words that contain numbers.
        # Remove words that are only one character.
        # Remove stopwords
        document = [token for token in document if not token.isnumeric()]
        document = [token for token in document if len(document) > 1]   
        document = [word for word in document if not word in self.stop_words]
        document = " ".join(document)

        # expand contractions    
        document = self.expand_contractions(document)

        # lemmatize
        document = document.split()
        lemmatizer = WordNetLemmatizer()
        document = [lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in document]
        document = [word for word in document if not word in self.stop_words]
        document = " ".join(document)

        return document
    

    ################## Topiuc Modelling Formatted output ##################

    def format_topics_sentences(self, ldamodel=None, corpus=None, texts=None):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list            
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df
    
    
    ################## Clean / Structure the Transcripts ##################

    def struct_file(self, tran):
        transcript = Transcript()
        doc_ana = []
        name_stp_wrds = []
        for p in range(0,len(tran)):
            transcript.parse(tran.iloc[p])
            for i in transcript.section_list:
                if not((i['section_type'] == 'Q&A') and (i['type'] == 'q')):
                    temp = {}
                    temp['comp_id'] = str(transcript.companies)
                    temp['time'] = transcript.title
                    name_stp_wrds.append(i['speaker'].lower().split())
                    temp['text'] = " ".join(j for j in i['text'])
                    doc_ana.append(temp)

        self.stop_words = self.stop_words.union(set(self.flatten(name_stp_wrds)))
        abc = pd.DataFrame(doc_ana)
        abc.columns = ['Company_ID', 'Time_Frame', 'Raw_Text']
        self.struct_df = abc
        

    ################## Subset Transcripts as per Company ID ##################

    def subset_trans(self, id, cutoff=5):
#         if stop_words==None:
#             stop_words = []
#         else:
#             stop_words = stop_words
        p = "['" + str(id) + "']"
        abc = self.struct_df
        abc1 = abc[abc.Company_ID == p]
        abc1 = abc1[abc1['Time_Frame'].str.contains('Q1|Q2|Q3|Q4')]
        abc1.reset_index(drop = True)
        abc1['Cleaned_Text'] = abc1['Raw_Text'].swifter.apply(lambda x: self.pre_process_document(x))
        abc1['clean_text_len'] = abc1['Cleaned_Text'].swifter.apply(lambda x: len(re.findall(r'\w+', x)))
        abc1['Year'] = abc1.Time_Frame.str.slice(3,7) + " " + abc1.Time_Frame.str.slice(0,2)
        abc1['Year2'] = abc1.Time_Frame.str.slice(3,7)
        abc1 = abc1[abc1.clean_text_len >= cutoff]
        abc1['Company_ID'] = abc1['Company_ID'].swifter.apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x))
        abc1 = abc1.sort_values(by='Year2', ascending=True)
        abc1.reset_index(drop=True, inplace=True)
        uniquetimes1, time_slices1 = np.unique(abc1.Year2, return_counts=True) 
        self.subset_df = abc1
        self.uniquetimes = uniquetimes1
        self.time_slices = time_slices1
        


    ################## Defining Topic models for sequences with optimal number of Topics ##################

    def compute_coherence_values(self, limit=11, start=4, step=1):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        limit : Max num of topics
        start : Least num of topics
        step  : Step-size

        Returns:
        -------
        model_list : List of LDA topic models (not now)
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics (not now)
        optim_k : optimal number of Topics
        """
        coherence_values = []
        perplexity_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamulticore.LdaMulticore(corpus=self.corpus,
                                                   id2word=self.dictionary,
                                                   num_topics=num_topics, 
                                                   passes=10,
                                                   alpha = 'asymmetric',
                                                   eta = 'auto',
                                                   random_state=42,
                                                   iterations = 500,
                                                   per_word_topics=True,
                                                   eval_every=None)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence()) 
            perplexity_values.append(model.log_perplexity(self.corpus))

        plot_val = [-1 * i / j for i, j in zip(coherence_values, perplexity_values)]
        x = np.array(coherence_values)
        z = (x-min(x))/(max(x)-min(x))
        scaled_coherence = z.tolist()
        scaled_subset = [i for i in scaled_coherence if i >= max(scaled_coherence)*0.7]
        scaled_subset_index = [scaled_coherence.index(i) for i in scaled_subset]

        try:
            scaled_subset_index.remove(4)
            if(len(scaled_subset_index)==0):
                scaled_subset_index = [4]    
        except:
            scaled_subset_index = scaled_subset_index
        finally:
            best_model = model_list[min(scaled_subset_index)]

        optim_k = best_model.get_topics().shape[0]

        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

        x = range(start, limit, step)
        plt.plot(x, perplexity_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Perplexity score")
        plt.legend(("perplexity_values"), loc='best')
        plt.show()
        
        self.num_topics = optim_k


    ################## Defining Topic models evaluation methods ##################

    def evaluate_model(self, coherence='c_v', seq=False):
        if(seq==True):
            coh_t = []
            for i in range(len(self.uniquetimes)):
                topics_dtm = self.model.dtm_coherence(time=i)
                temp_mod = CoherenceModel(topics=topics_dtm, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
                coh_t.append(temp_mod.get_coherence())

            avg_coherence = sum(coh_t) / len(coh_t)
            print('Average Model coherence: %.4f.' % avg_coherence)
            
            plt.plot(self.uniquetimes, coh_t)
            plt.xticks(rotation=70)
            plt.xlabel("Time Slices")
            plt.ylabel("Coherence score")
            plt.legend(("coherence_values"), loc='best')
            plt.show()

            fin = []
            cols = ["Topic-" + str(i+1) for i in range(len(self.model.doc_topics(0)))]
            for i in range(len(texts)):
                temp1 = self.model.doc_topics(i)
                fin.append(temp1)
                
            top_share = pd.DataFrame(fin, columns=cols)
            abc = pd.concat([self.time_df, top_share], axis=1)
            top_share = abc.groupby('Year2').mean().sort_values('Year2', ascending=True)

            for i in range(len(self.model.doc_topics(0))):
                y = np.array(top_share[cols[i]])
                plt.plot(self.uniquetimes, y, label="Topic-" + str(i+1))
            
            plt.xticks(rotation=70)
            plt.xlabel("Time Slices")
            plt.ylabel("Coherence score")
            plt.legend()
            plt.show()

        else:
            coherencemodel = CoherenceModel(model=self.model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            top_topics = self.model.top_topics(corpus=self.corpus, texts=self.texts, dictionary=self.dictionary, window_size=None, coherence='c_v', topn=20)
            # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
            avg_topic_coherence = sum([t[1] for t in top_topics]) / len(top_topics)
            print('Average topic coherence: %.4f.' % avg_topic_coherence)
            print('Model coherence: %.4f.', coherencemodel.get_coherence())
            print('Perplexity: ', self.model.log_perplexity(self.corpus))
            
            
    ################## Topic Trends for DTM ##################        
            
    def topic_trends(self, explore_topic=1, top_terms=5, normalize=False):
        result = pd.DataFrame([], columns = ['keywords'])
        for i in range(len(self.uniquetimes)):
            x = pd.DataFrame(self.model.print_topic(explore_topic-1, time=i, top_terms=top_terms), columns = ['keywords', 'weight'])
            if(normalize==True):
                x['Nmlz_Prob_'+ str(self.uniquetimes[i])] = x['weight']/x['weight'].sum()
            else:
                x['Nmlz_Prob_'+ str(self.uniquetimes[i])] = x['weight']
            x = x.drop(['weight'], axis=1)
            result = pd.merge(result, x, on='keywords', how='outer')

        for j in range(len(result)):
            y = np.array(result.loc[j])[1:]
            plt.plot(self.uniquetimes, y, label=str(result['keywords'][j]))
        
        plt.xticks(rotation=70)
        plt.xlabel("Time Slices")
        plt.ylabel("Trend Importance")
        plt.legend()
        plt.show()
        return result
        
        
    ################## Defining Topic models Training Method ##################

    def dynamic_model(self, n_top=None, timestamp=None, seq=False):
        """
        Function to define both Dynamic / Static topic models.

        Returns:

        lda_sta_<timestamp>.html = The pyLDA vis html to visualize the static LDA model
        lda_seq_snap_<timestamp>.html = The pyLDA vis html to visualize the snap at time for the sequential LDA model

        model_sta.pkl = The pickle object for static LDA model
        model_seq.pkl = The pickle object for static LDA model


        Parameters:

        n_top = (str or int) number of topics (optional) or else the default is 'optimal'

        timestamp = (str or int) To subset the data by value of 'Year2' column for static LDA and also 
                    the time-snapshot for LDA vis html in sequential model (for validation) as per values of Year2

        seq = (bool) To decide if builing a LDA Sequential Model

        """
        
        abc1 = self.subset_df
        if(seq==True):
            abc1 = abc1.sort_values(by='Year2', ascending=True)
            abc1.reset_index(drop=True, inplace=True)
            self.time_df = pd.DataFrame(abc1.Year2, columns=['Year2'])
            uniquetimes, time_slices = np.unique(abc1.Year2, return_counts=True)
            self.uniquetimes = uniquetimes
            self.time_slices = time_slices

            doc_lst = abc1.Cleaned_Text.tolist()
            doc_lst2 = []
            doc_lst = [nltk.word_tokenize(str(doc)) for doc in doc_lst]

            # Compute bigrams.
            # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
            bigram = Phrases(doc_lst, min_count=2, threshold = 1.0)
            for idx in range(len(doc_lst)):
                temp_bigram = []
                for token in bigram[doc_lst[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        temp_bigram.append(token)
                doc_lst.append(temp_bigram)

            # Create Corpus
            # data_lemmatized = [nltk.word_tokenize(str(sent)) for sent in doc_lst2]
            dictionary = corpora.Dictionary(doc_lst)
            dictionary.filter_extremes(no_below=2, no_above=0.8)
            corpus = [dictionary.doc2bow(text) for text in doc_lst]
            
            self.texts = doc_lst
            self.dictionary = dictionary
            self.corpus = corpus

            if (n_top=='optimal'):
                self.compute_coherence_values(limit=11, start=4, step=1)

            # Build LDA Sequence model
            ldamodel = gensim.models.ldaseqmodel.LdaSeqModel(corpus=self.corpus, time_slice=self.time_slices, id2word=self.dictionary,
                                                             alphas=0.01, num_topics=self.num_topics, initialize='gensim', sstats=None, 
                                                             lda_model=None, obs_variance=0.5, chain_variance=0.005, 
                                                             passes=10, random_state=42, lda_inference_max_iter=25, 
                                                             em_min_iter=6, em_max_iter=20, chunksize=100)

            self.ldamodel = ldamodel
            ldamodel.save(str(self.company) + "_model_seq.pkl")
            self.evaluate_model(coherence='c_v', seq=True)
            self.topic_trends(explore_topic=1, top_terms=5, normalize=False)

            # Visualize the topics
            pyLDAvis.enable_notebook()

            if (timestamp is not None):
                snap = uniquetimes.tolist().index(timestamp)
#                 print('Coherence at ' + timestamp + ': ', ldamodel.dtm_coherence(snap))
            else:
                snap = 0

            doc_topic, topic_term, doc_lengths, term_frequency, vocab = self.ldamodel.dtm_vis(time=snap, corpus=self.corpus)
            vis = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, 
                                       term_frequency=term_frequency, sort_topics = False)
            pyLDAvis.save_html(vis, str(self.company) + '_lda_seq_snap_' + str(uniquetimes[snap]) + '.html')

        else:
            doc_lst = abc1[abc1.Year2 == timestamp].Cleaned_Text.tolist()
            doc_lst2 = []
            doc_lst = [nltk.word_tokenize(str(doc)) for doc in doc_lst]

            # Compute bigrams.
            # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
            bigram = Phrases(doc_lst, min_count=2, threshold = 1.0)
            for idx in range(len(doc_lst)):
                temp_bigram = []
                for token in bigram[doc_lst[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        temp_bigram.append(token)
                doc_lst.append(temp_bigram)

            # Create Corpus
            # data_lemmatized = [nltk.word_tokenize(str(sent)) for sent in doc_lst2]
            dictionary = corpora.Dictionary(doc_lst)
            dictionary.filter_extremes(no_below=2, no_above=0.8)
            corpus = [dictionary.doc2bow(text) for text in doc_lst]

            self.texts = doc_lst
            self.dictionary = dictionary
            self.corpus = corpus
        
            if (n_top=='optimal'):
                self.compute_coherence_values(limit=11, start=4, step=1)

            # Build LDA model
            ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10, alpha='asymmetric', eta='auto', 
                                                               random_state=42, iterations = 500, per_word_topics=True, eval_every=None)

    #         ldamodel.print_topics()
            self.ldamodel = ldamodel
            ldamodel.save(str(self.company) + "_model_sta_" + str(timestamp) + ".pkl")
            self.evaluate_model(coherence='c_v', seq=False)

            # Visualize the topics
            pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics = False)
            pyLDAvis.save_html(vis, str(self.company) + '_lda_sta_' + str(timestamp) + '.html')

        return vis