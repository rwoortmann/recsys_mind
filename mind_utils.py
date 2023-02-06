import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   


import os
import pandas as pd
import zipfile
#import gdown
import csv
import json



# The zipfile contains all folds from the MIND dataset: msnews.github.io
# As well as scraped articles using their tool

def download_mind(file_id='1lDs6la081AiIMnX7rfEbqGjkO9nhhga6', data_dir='/dbfs/user/ruben.woortmann@persgroep.net/mind'):
    zip_path = '/tmp/mind_data.zip'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)  

        
# The behaviors.tsv file contains the impression logs and users' news click histories. 
# It has 5 columns divided by the tab symbol:
# - Impression ID. The ID of an impression.
# - User ID. The anonymous ID of a user.
# - Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
# - History. The news click history (ID list of clicked news) of this user before this impression.
# - Impressions. List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click).

def get_behaviors(set_dir):
    path = os.path.join(set_dir, 'behaviors.feather')
    if os.path.exists(path):
        df = pd.read_feather(path)
    else:
        df = pd.read_table(os.path.join(set_dir, 'behaviors.tsv'),
                                   header=None,
                                   #parse_dates=[2], 
                                   names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        
        df['pos'], df['neg'] = zip(*df.impressions.apply(split_impression_clicks))
        #df = df.sort_values(by=['time']).reset_index(drop=True)
        #df.to_feather(path)
    return df



# The news.tsv file contains the detailed information of news articles involved in the behaviors.tsv file.
# It has 7 columns, which are divided by the tab symbol:
# - News ID
# - Category
# - Subcategory
# - Title
# - Abstract
# - URL
# - Title Entities (entities contained in the title of this news)
# - Abstract Entities (entities contained in the abstract of this news)

def get_news(set_dir):
    news_path = os.path.join(set_dir, 'news.tsv')
    news = pd.read_table(news_path,
                         quoting=csv.QUOTE_NONE,
                         header=None,
                         names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                'title_entities', 'abstract_entities'])
    news['title_entities'] = news['title_entities'].apply(lambda row: json.loads(row))
    return news


# The entity_embedding.vec file contains the 100-dimensional embeddings
# of the entities learned from the subgraph by TransE method.
# The first column is the ID of entity, and the other columns are the embedding vector values.

def get_entity_embedding(set_dir):
    entity_embedding_path = os.path.join(set_dir, 'entity_embedding.vec')
    entity_embedding = pd.read_table(entity_embedding_path, header=None)

    entity_embedding['vector'] = entity_embedding.iloc[:, 1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector']].rename(columns={0: "entity"})

    return entity_embedding


# The relation_embedding.vec file contains the 100-dimensional embeddings
# of the relations learned from the subgraph by TransE method.
# The first column is the ID of relation, and the other columns are the embedding vector values.

def get_relation_embedding(set_dir):
    relation_embedding_path = os.path.join(set_dir, 'relation_embedding.vec')
    relation_embedding = pd.read_table(relation_embedding_path, header=None)

    relation_embedding['vector'] = relation_embedding.iloc[:, 1:101].values.tolist()
    relation_embedding = relation_embedding[[0, 'vector']].rename(columns={0: "relation"})

    return relation_embedding


# The article_texts.json file contains the article texts from each url in news.tsv.
# The nid column is an article id parsed from the url.

def get_article_texts(set_dir):
    article_texts_path = os.path.join(set_dir, 'article_texts.json')
    article_texts = pd.read_json(article_texts_path)
    article_texts.columns = ['url_id', 'article']

    return article_texts



def get_entities(set_dir):
    news_df = get_news(set_dir)
    entity_embeddings = get_entity_embedding(set_dir)
    
    entity_list = news_df[['news_id', 'title_entities']].explode('title_entities').dropna()
    entity_list['title_entities'] = entity_list['title_entities'].apply(lambda row: row['WikidataId'])
    
    # remove entities without embeddings
    entity_list = entity_list[entity_list['title_entities'].isin(entity_embeddings['entity'])]
    
    return entity_list, entity_embeddings
  
  
def get_user_history(set_dir):
      df = get_behaviors(set_dir)
      user_hist = df[['user_id', 'history']].drop_duplicates().dropna()
      user_hist['history'] = user_hist['history'].str.split(' ') 
      return user_hist



###
# Scraped articles
###

import re
import nltk
import numpy as np
from cleantext import clean

def remove_extra_spaces(row):
    row = re.sub('  ', ' ', row)
    row = re.sub(' ,', ',', row)
    row = re.sub(' \.', '.', row)
    return row

  
# Process HTML divs into a single article text
def clean_article_text(article_texts):
    
    # merge list to string
    article_texts['article'] = article_texts['article'].str.join(' ')

    # remove artifact spaces before punctuation
    article_texts['article'] = article_texts['article'].map(remove_extra_spaces)
    
    # text cleaner: https://pypi.org/project/clean-text/
    article_texts['article'] = article_texts['article'].map(lambda row: 
                                                            clean(row, lower=False, no_line_breaks=True, no_urls=True, no_emails=True))
    
    # id article texts that contain a roman character
    valid = article_texts['article'].str.contains('[A-Za-z]')
    
    # split article into sentences
    nltk.download('punkt')
    article_texts['sentences'] = article_texts['article'].map(nltk.sent_tokenize)
    
    # a sentence must contain > 3 characters, can occur due to html parsing errors
    article_texts['sentences'] = article_texts['sentences'].map(lambda row: 
                                                                [sentence for sentence in row if len(sentence) > 3])

    # number of sentences in an article, for batch padding
    article_texts['n_sentences'] = article_texts['sentences'].map(len)
    
    return article_texts[valid]
  
    
# Merging news.tsv and article_texts
def merge_news_articles(news, article_texts):
    
    # parse url_id from url to match article_texts' url_id
    news['url_id'] = news.apply(
        lambda row: row.url.split('/')[-1].split('.')[-2], axis=1)

    news_articles = pd.merge(news, article_texts, on=['url_id', 'url_id'], how='left')
    
    # replace missing sentence counts with 0, as column dtype should be integer
    news_articles['n_sentences'] = news_articles['n_sentences'].fillna(0).astype(int)

    return news_articles
  
    
 # Save / load news_articles dataframe
def load_news_articles(set_dir, remove_empty=True, remake=False):
    news_articles_path = os.path.join(set_dir, 'news_articles.feather')

    if os.path.exists(news_articles_path) and not remake:
        news_articles = pd.read_feather(news_articles_path)
    else:
        news = get_news(set_dir)
        articles = get_article_texts(set_dir)
        clean_articles = clean_article_text(articles)
        news_articles = merge_news_articles(news, clean_articles)
        news_articles.to_feather(news_articles_path)
    
    if remove_empty:
        news_articles.dropna(subset=['title', 'abstract', 'article', 'sentences'], inplace=True)
        news_articles.reset_index(inplace=True)
    
    return news_articles   




###
# History
###

def map_history(set_dir, id_to_node_map):
    df = get_user_history(set_dir)
    df['user_id'] = df['user_id'].apply(lambda x: id_to_node_map['user'][x])
    df['history'] = df['history'].apply(lambda x: [id_to_node_map['news'][x_i] for x_i in x])
    return df

  
# impression logs show which articles were seen, split into read and non-read 
def split_impression_clicks(row):
    pos, neg = [], []
    for news_id in row.split(' '):
        if news_id[-1] == '0':
            neg.append(news_id[:-2])
        elif news_id[-1] == '1':
            pos.append(news_id[:-2])
    return pos, neg  
  
  
###
# Lookup
###
def get_df_entry(df, column, idx):
    return df.loc[df[column] == idx]
  
  
def get_no_hist_users(set_dir, id_to_node_map):
    df = get_behaviors(set_dir)
    users = df[df.history.isna()].user_id.unique()
    users = [id_to_node_map['user'][x] for x in users]
    return users




def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            print(n)
            print(p.grad)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])




