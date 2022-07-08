import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer 
from sklearn.metrics.pairwise import cosine_similarity
import pickle

ps=PorterStemmer()
#convert function 

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name']) 
    return L 
# convert3 function
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 
# to fetch directors name
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
# function to stem words

def stem(text):
    y=[]

    for i in text:
        y.append(ps.stem(i))
    return y

#function to recommend movies

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

# creating df

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

#merging in one df

movies=movies.merge(credits,on='title')

#genres #id #keywords #title #overview #cast # crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# dropping missing data
movies.dropna(inplace=True)

# calling convert function to obtain list of genres
movies['genres'] = movies['genres'].apply(convert)

## calling convert function to obtain list of keywords
movies['keywords'] = movies['keywords'].apply(convert)

#calling convert3 function to obtain list of actors
movies['cast'] = movies['cast'].apply(convert3)

# calling fetch director function to keep only director's name in crew
movies['crew'] = movies['crew'].apply(fetch_director)

# to create overview string to list
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# removing spaces from each entity in list in each column
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# concatenation of all column in one column i.e tag
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# creating new df with movie_id , title ,tags
new_df=movies[['movie_id','title','tags']]

#applying stemming
ps=PorterStemmer()
new_df['tags']=new_df['tags'].apply(stem)

#converting list of tags to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

#converting to lower case
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

# creating vectors of 5000 dimension and removing stop words
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()

# creating similarity score matrix
similarity = cosine_similarity(vectors)


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

