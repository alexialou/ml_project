generate samples for modeling of clerk prediction using trigrams data

variable of input folder:
sentences_dir

variable of output folder:
outpath

variable of reference data:
clerk_path
meta_path

command to run:
python3.6 trigrams_data.py

People should know before starting:
this code generates the data year by year. You need to manually combine all the seperate data by year to one file, before giving this file to the modeling code.

Tutorial:
run this code to generate data for year 2010.
python3.6 trigrams_data.py
