from dotenv import load_dotenv
from annoy import AnnoyIndex
import pandas as pd
import numpy as np
import cohere
import os
import plotly.express as px
import umap
import plotly.graph_objects as go


def get_key():
    load_dotenv()
    return os.getenv("COHERE_API_KEY")


def import_ds():
    newsfiles = ['amharic','hausa','swahili','yoruba','igbo']
    
    df_am =  pd.read_csv(f'{newsfiles[0]}.csv')
    df_am = df_am.sample(frac=0.5)
    #df_en =  pd.read_csv(f'{newsfiles[1]}.csv')
    #df_en = df_en.sample(frac=0.3)
    df_hs =  pd.read_csv(f'{newsfiles[1]}.csv')
    df_hs = df_hs.sample(frac=0.5)
    df_sw =  pd.read_csv(f'{newsfiles[2]}.csv')
    df_sw = df_sw.sample(frac=0.5)
    df_yr =  pd.read_csv(f'{newsfiles[3]}.csv')
    df_yr = df_yr.sample(frac=0.5)
    df_ig =  pd.read_csv(f'{newsfiles[4]}.csv')
    df_ig = df_ig.sample(frac=0.5)
    
    df_news = pd.concat([df_am,df_hs,df_sw,df_yr,df_ig],axis=0)
    
    df_news = df_news.sample(frac = 1)
    
    df_news = df_news[df_news['title'].notna()]
    
    df_news = df_news.drop_duplicates("title")
        
    df_news  = df_news.sample(500)
     
    return df_news

    
def getEmbeddings(co,df):
    
    df['text'] = df['title'] + df['summary']
    
    df = df.drop(['title','id','summary'],axis=1)
    
    embeds = co.embed(texts=list(df['text']),model="multilingual-22-12",truncate="RIGHT").embeddings  
    
    embeds = np.array(embeds)
    
    return embeds

def semantic_search(emb,indexfile):
    
    emb = np.array(emb)

    search_index = AnnoyIndex(emb.shape[1], 'angular')
    print(emb.shape[1])

    for i in range(len(emb)):
        search_index.add_item(i, emb[i])

    search_index.build(10)
    search_index.save(indexfile)

def get_query_embed(co, query):
    query_embed = co.embed(texts=[query],
                           model='multilingual-22-12',
                           truncate='right').embeddings

    return np.array(query_embed)
    
def getClosestNeighbours(indexfile,query_embed,neighbours=15):

    search_index = AnnoyIndex(768, 'angular')
    search_index.load(indexfile)


    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],neighbours,
                                                        include_distances=True)
    
    return similar_item_ids

def display_news(df,similar_item_ids):
    # Format the results
    #print(similar_item_ids)
    results = pd.DataFrame(data={'title': df.iloc[similar_item_ids[0]]['title'],
                                 'url': df.iloc[similar_item_ids[0]]['url'],
                                  'summary': df.iloc[similar_item_ids[0]]['summary']})
                                 #'distance': similar_item_ids[1]})
    results.reset_index(drop=True, inplace=True)                            
    
    return results

def getUMAPEmbed(embeds):
    # Map the nearest embeddings to 2d
    reducer = umap.UMAP(n_neighbors=20)
    
    return reducer.fit_transform(embeds)


def plot2DChart(df, umap_embeds, clusters=None):
    if clusters is None:
        clusters = {}

    df_explore = pd.DataFrame(data={'url': df['url'], 'title': df['title']})
    df_explore['x'] = umap_embeds[:, 0]
    df_explore['y'] = umap_embeds[:, 1]

    print(df_explore)
    # Plot
    fig = px.scatter(df_explore, x='x', y='y', hover_data=['title'])

    for cluster in clusters.values():
        high_freq_words = str(list({x: count for x, count in cluster[2].items() if count >= 3}.keys())[:10])
        fig.add_trace(go.Scatter(
            x=cluster[0],
            y=cluster[1],
            fill="toself",
            mode='lines',
            text=high_freq_words,
            opacity=0.5,
            showlegend=False

        ))



    fig.data = fig.data[::-1]

    return fig

if __name__ == '__main__':
    key = get_key()
    co = cohere.Client(key)
    df_news = import_ds()
    embed = process(co,df_news)
    semantic_search(embed)
    getClosestNeighbours(df_news)