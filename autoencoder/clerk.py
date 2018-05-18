
# coding: utf-8

# In[1]:


import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import os
import glob
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from ast import literal_eval
import sdae
import timeit
import theano
import sys
from sklearn.model_selection import train_test_split
import scipy


# In[2]:

os.chdir("/scratch/cg3015/clerks")
sentences_dir = 'bigrams_sentences/'
fw_pkl = '/scratch/cg3015/clerks/function_words.pkl'
fws = pickle.load(open(fw_pkl, 'rb'))
clerk_path = 'Circuit-Courts_1995_2016__31jan2018.xls'
meta_path = 'circuit_metadata_excerpt.dta'
#id_songer_path = 'data/originalism/caseid_songername.dta'
outpath = 'data/'
def func_word_freq(raw_txt, fws):
    raw = open(raw_txt, 'r').read()
    # fws = pickle.load(open(fw_pkl, 'rb'))
    stemmer = SnowballStemmer("english")
    tokens = word_tokenize(raw)
    stemmed = [stemmer.stem(t) for t in tokens]
    word_count = Counter(stemmed)
    res = {}
    for k, v in word_count.items():
        if k in fws:
            res[k] = v
    return res


# In[3]:


def get_clerks(clerk_df, judge, year):
    res_clerk = clerk_df.loc[(clerk_df['Judge Name']==judge) & (clerk_df['Year']==year), 'Clerk Law School']
    return res_clerk


# In[4]:


def parse_sentences(sentences_dir, out_path, clerk_path, meta_path, fws, case_year=-1):
    clerk_df = pd.read_excel(clerk_path, sheet_name = 1)
    meta_df = pd.read_stata(meta_path)
    
    if case_year == -1:
        files = sentences_dir + '*/*.txt'
    else:
        files = sentences_dir + 'sent_' + repr(case_year) + '/*.txt'
    cols = ['year', 'judge', 'fw_count', 'clerk_school']
    tokenizer = RegexpTokenizer(r'\w+')
    data_list = []
    for filename in glob.iglob(files, recursive=True):
        caseid = filename.split('/')[-1].split('_')[0]
        songer_names = meta_df.loc[(meta_df['caseid']==caseid) & 
                                   (meta_df['j']==meta_df['Writer']) &
                                   (meta_df['songername']!=''),'songername']
        if songer_names.count() > 0:
            if songer_names.count() > 1:
                print("more than 1 entry for case: "+ caseid)
            songer_name = songer_names.values[0]
            songer_name_split = tokenizer.tokenize(songer_name)
            songer_name_split.sort(key=len, reverse=True)
            first_last = songer_name_split[:2]
            if(len(songer_name_split)<2):
                print(songer_name_split)
                print(caseid)
            judge_clerk_df = clerk_df.loc[(clerk_df['Year']==case_year) & 
                                          (clerk_df['Clerk Law School'].notnull()) &
                                          (clerk_df['Judge Name'].str.contains(first_last[0].title())) & 
                                          (clerk_df['Judge Name'].str.contains(first_last[1].title())), 
                                          ['Judge Name', 'Clerk Law School']].drop_duplicates()
            for idx, row in judge_clerk_df.iterrows():
                entry = [case_year, row['Judge Name'], func_word_freq(filename, fws), row['Clerk Law School']]
                data_list.append(entry)
    parsed_data = pd.DataFrame(data = data_list, columns = cols)
    if case_year != -1:
        parsed_data.to_csv(out_path + repr(case_year) + '_parsed.csv', index=False)
    else:
        parsed_data.to_csv(out_path + 'parsed.csv', index=False)


# In[ ]:


def clean_parsed(parsed_csv):
    df = pd.read_excel(parsed_csv)
    df['Clerk Law School'] = df['Clerk Law School'].replace(['Stanford_1993'],'Stanford')
    df['Clerk Law School'] = df['Clerk Law School'].replace(['UC Berkeley'],'Berkeley')
    df['Clerk Law School'] = df['Clerk Law School'].str.replace('_',' ')
    return df


# In[22]:

def convert(doc):
    dict = {}
    doc = doc.replace('[', '')
    doc = doc.replace(']', '')
    doc = doc.replace(' ', '')
    doc = doc.split(',')
    doc = list(map(lambda x : int(x), doc))

    for i in range(len(doc)):
        if doc[i] not in dict:
            dict[doc[i]] = 0
        dict[doc[i]] += 1

    res = "\"{"
    for k, v in dict.items():
        res += ("'" + str(k) + "': " + str(v) + ",")
    res = res[0:-1]
    res += "}\""
    return res 

def feature_norm(cleaned_csv, isFolder=False):
    if isFolder:
        allFiles = glob.glob(cleaned_csv + '*.csv')
        parsed = pd.DataFrame()
        list_ = []
        for file_ in allFiles:
            df = pd.read_csv(file_)
            list_.append(df)
            parsed = pd.concat(list_, ignore_index=True)
    else:
        parsed = pd.read_csv(cleaned_csv)
    fw_count = parsed['fw_count']
    #for index, row in fw_count.items():
    #    fw_count[index] = convert(row)
    #fw_count = fw_count.apply(convert)
    fw_count = fw_count.tolist()
    fw_count = list(map(lambda fw: literal_eval(fw), fw_count))
    v = DictVectorizer(sparse=False)
    #print("eval done")
    #v = DictVectorizer()
    fw_vec = v.fit_transform(fw_count)
    fw_nomed = normalize(fw_vec, axis=0, norm='max')
    print(v.feature_names_)
    fw_df = pd.DataFrame(data = fw_nomed, columns = v.feature_names_)
    expanded_df = pd.concat([parsed, fw_df], axis=1)
    expanded_df.drop('fw_count', axis=1, inplace=True)
    expanded_df.to_csv('data/cleaned.csv', index=False)


# In[36]:


# df = pd.read_csv('data/cleaned.csv')
# df['clerk_school'] = pd.Categorical(df.clerk_school).codes
#y_data = df['clerk_school'].as_matrix()

#print(np.unique(y_data))
#print(len(df.columns))


# In[ ]:
#parse_sentences(sentences_dir, outpath, clerk_path, meta_path, fws, 2012)
#feature_norm('data/2012_parsed.csv')
#data generated 
#feature_norm('data/1995_trigrams_judge_clerk.csv')
#feature_norm('data/4year_trigrams_judge_clerk.csv')
feature_norm('data/all_trigrams.csv')
normed = pd.read_csv('data/cleaned.csv')
normed = normed.loc[normed['clerk_school'].notnull()]
normed = normed.drop(['judge'], axis=1)
normed['clerk_school'] = pd.Categorical(normed.clerk_school).codes
y_data = normed['clerk_school'].as_matrix()
X_data = normed.drop(['clerk_school'], axis=1).as_matrix()

uniq_sch = len(np.unique(y_data))
print(X_data.shape)
feature_num = X_data.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = theano.shared(X_train.astype('float64'))
y_train = theano.shared(y_train.astype('int32'))
X_val = theano.shared(X_val.astype('float64'))
y_val = theano.shared(y_val.astype('int32'))
X_test = theano.shared(X_test.astype('float64'))
y_test = theano.shared(y_test.astype('int32'))

datasets = np.array([(X_train, y_train), (X_val, y_val), (X_test, y_test)])


finetune_lr=0.1
pretraining_epochs=3
pretrain_lr=0.001,
training_epochs=1000
batch_size=1

X_train = datasets[0][0]
n_train_batches = X_train.get_value(borrow=True).shape[0]
n_train_batches //= batch_size

numpy_rng = np.random.RandomState(89677)
encoder = sdae.SdA(numpy_rng=numpy_rng,
                   n_ins=feature_num,
                   hidden_layers_sizes=[1000, 1000, 1000],
                   n_outs=uniq_sch)
pretraining_fns = encoder.pretraining_functions(train_set_x=X_train, batch_size=batch_size)
print('... pre-training the model')

start_time = timeit.default_timer()
## Pre-train layer-wise
corruption_levels = [.1, .2, .3]
for i in range(encoder.n_layers):
    # go through pretraining epochs
    for epoch in range(pretraining_epochs):
        # go through the training set
        c = []
        for batch_index in range(n_train_batches):
            c.append(pretraining_fns[i](index=batch_index, corruption=corruption_levels[i]))
#             c.append(pretraining_fns[i](index=batch_index,
#                                         corruption=corruption_levels[i],
#                                         lr=pretrain_lr))
                
        print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c, dtype='float64')))
end_time = timeit.default_timer()
print(('The pretraining code ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

print('... getting the finetuning functions')
train_fn, validate_model, test_model = encoder.build_finetune_functions(
    datasets=datasets,
    batch_size=batch_size,
    learning_rate=finetune_lr
)
print('... finetunning the model')
    # early-stopping parameters
patience = 100 * n_train_batches  # look as this many examples regardless
patience_increase = 2.  # wait this much longer when a new best is
                            # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

best_validation_loss = np.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0

while (epoch < training_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_fn(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            validation_losses = validate_model()
            this_validation_loss = np.mean(validation_losses, dtype='float64')
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if (this_validation_loss < best_validation_loss * improvement_threshold):
                    patience = max(patience, iter * patience_increase)
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
            if patience <= iter:
                done_looping = True
            break

end_time = timeit.default_timer()
print((
        'Optimization complete with best validation score of %f %%, '
        'on iteration %i, '
        'with test performance %f %%'
    ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print(('The training code ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



# In[114]:


'''
https://stackoverflow.com/questions/4771293/can-an-authors-unique-literary-style-be-used-to-identify-him-her-as-the-autho
function words: 
    http://www.sequencepublishing.com/1/academic.html
    https://stackoverflow.com/questions/5819840/calculate-frequency-of-function-words?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    
'''


# In[ ]:


'''
serv_sentences_dir = '/data/Dropbox/Projects/Ash_Chen/clerkships/sentences_new/'
serv_fw_pkl = '/data/Dropbox/Projects/Ash_Chen/clerkships/function_words.pkl'
serv_fws = pickle.load(open(serv_fw_pkl, 'rb'))
serv_clerk_path = '/data/Dropbox/Projects/Ash_Chen/clerkships/Circuit-Courts_1995_2016__31jan2018.xls'
serv_outpath = '/data/Dropbox/Projects/Ash_Chen/clerkships/'
serv_meta_path = '/data/Dropbox/Projects/Ash_Chen/clerkships/circuit_metadata_excerpt.dta'
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws)
'''

