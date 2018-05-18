import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import os 
import pickle
import pandas as pd
from datetime import datetime
#from nltk.tokenize.regexp import RegexpTokenizer
import glob

directory = "/data/Dropbox/Projects/Ash_Chen/clerkships/trigrams_NYU_clerk"
os.chdir(directory)
sentences_dir = directory + '/input/phrasefreqs_txt/'
outpath = '/data/Dropbox/Projects/Ash_Chen/clerkships/trigrams_NYU_judge/output/'
clerk_version = 1

if clerk_version == 0:
    clerk_path = '/data/Dropbox/Projects/Ash_Chen/clerkships/Circuit-Courts_1995_2016__31jan2018.xls'
elif clerk_version == 1:
    clerk_path = '/data/Dropbox/Projects/Ash_Chen/clerkships/trigrams_NYU_clerk/ref/clerkship_1995_2016_merged_martindale.csv' 

meta_path = '/data/Dropbox/Projects/Ash_Chen/clerkships/circuit_metadata_excerpt.dta'
fws = None


def feature_two_grams(sentences_dir, out_path, clerk_path, meta_path, fws, case_year=-1):
    if clerk_version == 0:
        clerk_df = pd.read_excel(clerk_path, sheet_name = 1)
    elif clerk_version == 1:
        clerk_df = pd.read_csv(clerk_path, sep = ',', encoding = 'ISO-8859-1')
    meta_df = pd.read_stata(meta_path)

    if case_year == -1:
        files = sentences_dir + '*/*.txt'
    else:
        #files = sentences_dir + repr(case_year) + '/*.pkl.pkl'
        files = sentences_dir + 'sent_' + repr(case_year) + '/*.txt'
    print("data loaded....")        
    #cols = ['year', 'judge', 'fw_count', 'clerk_school']
    cols = ['year', 'fw_count', 'judge']
    tokenizer = RegexpTokenizer(r'\w+')
    data_list = []
    for filename in glob.iglob(files, recursive=True):
        break
        caseid = filename.split('/')[-1].split('_')[0]
        songer_names = meta_df.loc[(meta_df['caseid']==caseid) & 
                                   (meta_df['j']==meta_df['Writer']) &
                                   (meta_df['songername']!=''),'songername']
        if songer_names.count() > 0:
            if songer_names.count() > 1:
                print("more than 1 entry for case: "+ caseid)
            songer_name = songer_names.values[0]
            '''
            songer_name_split = tokenizer.tokenize(songer_name)
            songer_name_split.sort(key=len, reverse=True)
            first_last = songer_name_split[:2]
            if(len(songer_name_split)<2):
                print(songer_name_split)
                print(caseid)
            if clerk_version == 0:
                judge_clerk_df = clerk_df.loc[(clerk_df['Year']==case_year) & 
                                          (clerk_df['Clerk Law School'].notnull()) &
                                          (clerk_df['Judge Name'].str.contains(first_last[0].title())) & 
                                          (clerk_df['Judge Name'].str.contains(first_last[1].title())), 
                                          ['Judge Name', 'Clerk Law School']].drop_duplicates()
            elif clerk_version == 1:
                judge_clerk_df = clerk_df.loc[(clerk_df['year_of_clerkship'] == case_year) &
                                          (clerk_df['ClerkLawSchool'].notnull()) &
                                          (clerk_df['JudgeName'].str.contains(first_last[0].title())) &
                                          (clerk_df['JudgeName'].str.contains(first_last[1].title())),
                                          ['JudgeName', 'ClerkLawSchool']].drop_duplicates()
            '''
            file = open(filename, 'rb')
            tri_grams = pickle.load(file)
            '''
            for idx, row in judge_clerk_df.iterrows():
                if clerk_version == 0:
                    entry = [case_year, row['Judge Name'], two_grams, row['Clerk Law School']]
                elif clerk_version == 1:
                    entry = [case_year, row['JudgeName'], two_grams, row['ClerkLawSchool']]
                data_list.append(entry)
            '''
            entry = [case_year, tri_grams, songer_name]
            data_list.append(entry)
    parsed_data = pd.DataFrame(data = data_list, columns = cols)
    if case_year != -1:
        parsed_data.to_csv(out_path + repr(case_year) + '_parsed_trigrams.csv', index=False)
    else:
        parsed_data.to_csv(out_path + 'parsed_judge.csv', index=False)
    return parsed_data

for year in range(1995, 2014):
    df = feature_two_grams(sentences_dir, outpath, clerk_path, meta_path, fws, year)
