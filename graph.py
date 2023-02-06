from collections import defaultdict
from mind_utils import *
import torch

class DataGraph(object):
    def __init__(self, set_dir, text_enc, attributes, device):
        
        self.atr_nodes = attributes
        self.device=device
        
        self.mapping = {}
        self.features = {}
        self.edge_index = {}

        news_df = self.get_news(set_dir)
        hist_df = self.get_user_history(set_dir)

        self.mapping['news'] = self.map_col(news_df.news_id)
        self.mapping['users'] = self.map_col(hist_df.user_id)
        
        self.features['news'] = self.get_news_x(text_enc, news_df).to(device)
        self.load_edge_index(hist_df.explode('history'), 'history', 'user_id', 'users')
       
        
        if 'category' in attributes:
            self.mapping['category'] = self.map_col(news_df.category)
            self.load_edge_index(news_df[['news_id', 'category']], 'news_id', 'category', 'category')
            
        if 'entities' in attributes:
            entity_list, entity_embeddings = self.get_entities(set_dir)
            self.mapping['entities'] = self.map_col(entity_embeddings.entity)
            self.features['entities'] = torch.tensor(entity_embeddings.vector.tolist()).to(device)
            self.load_edge_index(entity_list, 'news_id', 'title_entities', 'entities')
        
        if 'subcategory' in attributes:
            self.mapping['subcategory'] = self.map_col(news_df.subcategory)
            self.load_edge_index(news_df[['news_id', 'subcategory']], 'news_id', 'subcategory', 'subcategory')
        
        self.n_nodes = {key: len(self.mapping[key]) for key in self.mapping.keys()}
   
        
    def map_col(self, column):
        return {index: i for i, index in enumerate(column.unique())} 
         
         
    def load_edge_index(self, df, news_col, atr_col, atr_map):
        news_edge = [self.mapping['news'][index] for index in df[news_col]]
        atr_edge = [self.mapping[atr_map][index] for index in df[atr_col]]
 
        self.edge_index[atr_map] = torch.tensor([news_edge, atr_edge]).to(self.device)
        

    def get_news_x(self, text_enc, news_df):
        news_text = news_df['title'] + ' ' + news_df['abstract'].fillna('')
        news_x =  text_enc.encode(news_text.values)
        # remove zero'd columns
        news_x = news_x[:, news_x.std(dim=0) != 0.]
        # standardize
        news_x = (news_x - news_x.mean(dim=0)) / news_x.std(dim=0)
        return news_x
        
    def get_entities(self, set_dir):
        news_df = get_news(set_dir)
        entity_embeddings = get_entity_embedding(set_dir)

        entity_list = news_df[['news_id', 'title_entities']].explode('title_entities').dropna()
        entity_list['title_entities'] = entity_list['title_entities'].apply(lambda row: row['WikidataId'])

        # remove entities without embeddings
        entity_list = entity_list[entity_list['title_entities'].isin(entity_embeddings['entity'])]

        return entity_list, entity_embeddings
      
    def get_user_history(self, set_dir):
        df = get_behaviors(set_dir)
        user_hist = df[['user_id', 'history']].drop_duplicates().dropna(subset=['history'])
        user_hist['history'] = user_hist['history'].str.split(' ') 
        return user_hist


    def get_news(self, set_dir):
        news_path = os.path.join(set_dir, 'news.tsv')
        news = pd.read_table(news_path,
                            quoting=csv.QUOTE_NONE,
                            header=None,
                            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                    'title_entities', 'abstract_entities'])
        news['title_entities'] = news['title_entities'].apply(lambda row: json.loads(row))
        return news
