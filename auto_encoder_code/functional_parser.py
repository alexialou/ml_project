#
# this file read from the sentence_new dir and generate a csv file
# matching each file's functional word counts to its potential clerks
# the word counts are normalized
#


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
import sys
import scipy

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

def get_clerks(clerk_df, judge, year):
    res_clerk = clerk_df.loc[(clerk_df['Judge Name']==judge) & (clerk_df['Year']==year), 'Clerk Law School']
    return res_clerk

def parse_sentences(sentences_dir, out_path, clerk_path, meta_path, fws, case_year=-1):
    print("parsing year ", case_year)
    clerk_df = pd.read_csv(clerk_path, encoding='ISO-8859-1')
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
            judge_clerk_df = clerk_df.loc[(clerk_df['year_of_hire']==case_year) & 
                                          (clerk_df['ClerkLawSchool'].notna()) &
                                          (clerk_df['judge_name_robust'].str.contains(first_last[0].upper())) & 
                                          (clerk_df['judge_name_robust'].str.contains(first_last[1].upper())), 
                                          ['judge_name_robust', 'ClerkLawSchool']].drop_duplicates()
            for idx, row in judge_clerk_df.iterrows():
                entry = [case_year, row['judge_name_robust'], func_word_freq(filename, fws), row['ClerkLawSchool']]
                data_list.append(entry)
    parsed_data = pd.DataFrame(data = data_list, columns = cols)
    if case_year != -1:
        parsed_data.to_csv(out_path + repr(case_year) + '_parsed.csv', index=False)
    else:
        parsed_data.to_csv(out_path + 'parsed.csv', index=False)

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
    fw_count = parsed['fw_count'].tolist()
    fw_count = list(map(lambda fw: literal_eval(fw), fw_count))
    v = DictVectorizer(sparse=False)
    fw_vec = v.fit_transform(fw_count)
    fw_nomed = normalize(fw_vec, axis=0, norm='max')
    fw_df = pd.DataFrame(data = fw_nomed, columns = v.feature_names_)
    expanded_df = pd.concat([parsed, fw_df], axis=1)
    expanded_df.drop('fw_count', axis=1, inplace=True)
    expanded_df.to_csv(serv_outpath+'cleaned.csv', index=False)


# for local testing:
# sentences_dir = 'data/Circuit_Courts/circuit-cases/sentences_new/'
# fw_pkl = 'data/function_words.pkl'
# fws = pickle.load(open(fw_pkl, 'rb'))
# clerk_path = 'data/clerkships/clerkship_1995_2016_merged_martindale.csv'
# meta_path = 'data/originalism/data/circuit_metadata_excerpt.dta'
# id_songer_path = 'data/originalism/caseid_songername.dta'
# outpath = 'data/'
serv_sentences_dir = '/data/Dropbox/Projects/Ash_Chen/clerkships/sentences_new/'
serv_fw_pkl = '/data/Dropbox/Projects/Ash_Chen/clerkships/function_words.pkl'
serv_fws = pickle.load(open(serv_fw_pkl, 'rb'))
serv_clerk_path = '/data/Dropbox/Projects/Ash_Chen/clerkships/clerkship_1995_2016_merged_martindale.csv'
serv_outpath = '/data/Dropbox/Projects/Ash_Chen/clerkships/functional_NYU_clerk/'
serv_meta_path = '/data/Dropbox/Projects/Ash_Chen/clerkships/circuit_metadata_excerpt.dta'


parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 1995)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 1996)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 1997)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 1998)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 1999)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2000)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2001)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2002)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2003)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2004)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2005)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2006)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2007)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2008)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2009)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2010)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2011)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2012)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2013)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2014)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2015)
parse_sentences(serv_sentences_dir, serv_outpath, serv_clerk_path, serv_meta_path, serv_fws, 2016)

feature_norm(serv_outpath, isFolder=True)

'''
https://stackoverflow.com/questions/4771293/can-an-authors-unique-literary-style-be-used-to-identify-him-her-as-the-autho
function words: 
    http://www.sequencepublishing.com/1/academic.html
    https://stackoverflow.com/questions/5819840/calculate-frequency-of-function-words?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    
'''

