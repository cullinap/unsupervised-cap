from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bag_of_words():
	return CountVectorizer(analyzer = "word",   
                           tokenizer = None,    
                           preprocessor = None, 
                           stop_words = None,   
                           max_features = 5000
                        )

def tfidf():
	return TfidfVectorizer(max_df=0.5, 
                           min_df=2, 
                           stop_words='english', 
                           lowercase=True, 
                           use_idf=True,
                           norm=u'l2', 
                           smooth_idf=True 
                         )