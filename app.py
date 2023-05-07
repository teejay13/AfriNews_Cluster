import streamlit as st

from process import *

df = import_ds()

st.title('AFri News Multilingual Embedding')

form = st.form(key="user_settings")

textcontainer = st.container()

plotcontainer = st.container()

with form:

    query = st.text_input('Please input your news text here:')

    num_nearest = int(st.slider('Please input the number of news to find: ', value=15, min_value=1, max_value=200))
    
    generate_button = form.form_submit_button("Cluster News")

    if generate_button:
        key = get_key()

        co = cohere.Client(key)
        
        embeddings = getEmbeddings(co,df)

        indexfile = 'news.ann'

        semantic_search(embeddings, indexfile)

        query_embed = get_query_embed(co, query)

        nearest_ids = getClosestNeighbours(indexfile, query_embed, num_nearest)
        
        nn_embeddings = embeddings[nearest_ids[0]]
        
        all_embeddings = np.vstack([nn_embeddings, query_embed])
        
        umap_embeds  = getUMAPEmbed(embeddings)
        
        text_news = display_news(df,nearest_ids)
        
        fig = plot2DChart(df, umap_embeds)
        
        textcontainer.write(text_news)
        
        plotcontainer.write(fig)
        
