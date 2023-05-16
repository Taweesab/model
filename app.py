import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np
import random
from gensim import similarities
import pymongo

app=Flask(__name__)

@app.route('/recommendation', methods=['POST'])
def index():
    data = pd.read_csv('Nutrition.csv')
    regex = r'\d+'

    nutrition = request.json
    query = nutrition['food']
   
    ldamodel = load_file_from_pickle('lda_40.obj')

    dictionary = load_file_from_pickle('lda_40_dct.obj')

    corpus = load_file_from_pickle('lda_40_corp.obj')

    result = get_similarity_reco(query,ldamodel,dictionary,corpus,5)
    
    results_final = data.iloc[result]
    name = results_final['name'].replace(",", "")
    nameNew = ',' .join((z for z in name if not z.isdigit()))

    print(nameNew)
   
    return jsonify({'data':nameNew})

@app.route('/', methods=['GET'])
def hello():
    return "hello world"



def load_file_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        item = pickle.load(file)
    return item

def get_similarity(lda, query_vector,corpus):
        index = similarities.MatrixSimilarity(lda[corpus])
        sims = index[query_vector]
        return sims

    # define treat input function, returning a list of tokenized ingredients

def treat_words (words):
    list_words = words.split(",")
    output = []
    for w in list_words:
        output.append("_".join(w.strip().split(" ")))
    return output


def calculate_similarity(query,ldamodel,dct,corpus):
        # treat input words
        words_bow = dct.doc2bow(treat_words(query))
        query_vector = ldamodel[words_bow]
        
        # print(query_vector)
        #calculate ranking
        sim_rank = get_similarity(lda =ldamodel, query_vector = query_vector,corpus=corpus)
        sim_rank = sorted(enumerate(sim_rank), key=lambda item: -item[1])
        
        return [sim_rank,query_vector]

def calculate_recommendation(sim_rank,groups,ldamodel,query_vector,n_reco):
        results = []
        results_prob = []
        result_group = []
        result_name = []
        query_groups = ldamodel[query_vector]
        ## Find max score of tuple 
        max_score_tuple = max(query_groups, key=lambda tup: tup[1])
        
        # print("query_group",query_groups)
        # print("max_score_tuple",max_score_tuple)
        
        ## Find in same topic with max score
        for recipe,group in zip(sim_rank,groups):
            if group == max_score_tuple[0]:
                results.append(recipe[0])
                result_group.append(group)
                results_prob.append(recipe[1])
            if len(results) == n_reco:
                break
        # print(result_group,"\n",results_prob,"\n")
        return results

    # this is a wrapper function for calculate simu and calculate reco
def get_similarity_reco (query,ldamodel,dct,corpus,n_reco):
        #calculate rank
        [sim_rank,query_vector] = calculate_similarity(query,ldamodel,dct,corpus)
        #find groups according to lda model
        groups = []
        for l in ldamodel[corpus]:
            try:
                groups.append(l[0][0])
            except:
                groups.append(random.randint(1, 100))
                
        return calculate_recommendation(sim_rank,groups,ldamodel,query_vector,n_reco)

if __name__ == '__main__':
    # Start the Flask application
  app.run(debug=True,port=8080)