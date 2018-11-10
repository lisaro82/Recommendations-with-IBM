import pandas as pd
import numpy as np

def getArticles(p_articlesID, p_articles):
    return p_articles.set_index('article_id').loc[p_articlesID, ['title', 'topic_keywords']].reset_index()

def getTopArticlesID(p_userArticles, p_top = 20):
    v_data = p_userArticles.groupby(['article_id']).agg({'article_id': ['count']}).reset_index()
    v_data.columns = ['article_id', 'Article Read No']
    return v_data.sort_values('Article Read No', ascending = False).head(p_top)

def getTopArticles(p_articles, p_userArticles, p_top = 20):
    v_TopArticlesID = getTopArticlesID( p_userArticles = p_userArticles, 
                                        p_top          = p_top )
    v_TopArticlesID = v_TopArticlesID.merge(p_articles, how = 'inner', on = 'article_id')    
    return v_TopArticlesID[['article_id', 'title', 'topic_keywords', 'Article Read No']]

class User():
    
    __user_id__              = None
    __userArticles__         = None
    __userArticlesNo__       = None
    __userArticlesInteract__ = None
    
    def __init__(self, p_user_id, p_articles, p_userArticles, p_userArticlesMatrix):
        self.__user_id__ = p_user_id
        self.__setUserArticles__(p_articles, p_userArticles, p_userArticlesMatrix)
        return
    
    def getUserID(self):
        return self.__user_id__
    
    def getUserArticles(self):
        return self.__userArticles__
    
    def getUserArticlesNo(self):
        return self.__userArticlesNo__
    
    def getUserArticlesInteract(self):
        return self.__userArticlesInteract__
    
    def __setUserArticles__(self, p_articles, p_userArticles, p_userArticlesMatrix):
        v_userArticleGroup = p_userArticles.groupby(['article_id']).agg({'article_id': ['count']})
        v_userArticleGroup.columns = ['Article Read No']
        
        v_userArticlesMatrix = p_userArticlesMatrix.loc[self.__user_id__, :]
        v_userArticlesID = v_userArticlesMatrix[~v_userArticlesMatrix.isnull()].index.tolist()    
        
        self.__userArticlesNo__       = len(v_userArticlesID)
        self.__userArticlesInteract__ = ( p_userArticles[p_userArticles['user_id'] == self.__user_id__]
                                             .count().reset_index().loc[0, 0] )
        
        v_articles = getArticles( p_articlesID = v_userArticlesID,
                                  p_articles   = p_articles )

        v_articles = v_articles.merge(v_userArticleGroup.reset_index(), how = 'inner', on = 'article_id')
        v_articles['user_id'] = self.__user_id__
        
        self.__userArticles__ = v_articles.sort_values('Article Read No', ascending = False).reset_index(drop = True)
        return
    
    
class RecommendArticles():
    
    __User__             = None
    __recNumber__        = None # Number of articles to be recommended
    __artPerUser__       = None # Number of articles selected from a particular user read articles
    
    __keywords__         = None
    
    __similarUsersID__   = None
    __similarUsersDet__  = None
    
    __topArticles__      = None
    __recommendations__  = None
    
    def __init__(self, p_user_id, p_articles, p_userArticles, p_userArticlesMatrix, p_articleSimilarity,
                       p_artPerUser = 10, p_top = 30, p_keywords = [] ): 
        
        v_articles = p_articles.append(pd.DataFrame({ 'article_id':     -999, 
                                                      'topic_keywords': 'nan' }, index = [-1]), ignore_index = True, sort = True)
        
        self.__User__ = User( p_user_id            = p_user_id, 
                              p_articles           = v_articles, 
                              p_userArticles       = p_userArticles, 
                              p_userArticlesMatrix = p_userArticlesMatrix )
        self.__setSimilarUsers__( p_articles           = v_articles, 
                                  p_userArticles       = p_userArticles, 
                                  p_userArticlesMatrix = p_userArticlesMatrix,
                                  p_top                = round(p_top / 2) )
        self.__topArticles__ = getTopArticles( p_articles     = v_articles, 
                                               p_userArticles = p_userArticles, 
                                               p_top          = v_articles.shape[0] )
        
        self.__recNumber__  = p_top
        self.__artPerUser__ = p_artPerUser
        self.__keywords__   = p_keywords
        self.__getRecommendations__( p_articles           = v_articles, 
                                     p_articleSimilarity  = p_articleSimilarity )
        
        return
            
    def getUser(self):
        return self.__User__
            
    def getSimilarUsers(self):
        return self.__similarUsersID__
    
    def getSimilarUsersDetails(self, p_user_id):
        return self.__similarUsersDet__[p_user_id]
    
    def getRecommendations(self):
        return self.__recommendations__[['article_id', 'title', 'Keywords Similarity']]
    
    def getRecommendationsDetails(self):
        return self.__recommendations__
    
    def __getRecommendations__(self, p_articles, p_articleSimilarity): 
        def setKeywordsSimilarity(p_articles):
            if len(self.__keywords__) == 0:
                p_articles['Keywords Similarity'] = 0
                return p_articles
            
            p_articles['Keywords Similarity'] = 0
            for keyword in self.__keywords__:
                p_articles['Keywords Similarity'] = p_articles['Keywords Similarity'] \
                                                     + p_articles['topic_keywords'].apply(lambda x: 1 if keyword in x else 0)
            return p_articles
        
        def removeDuplicates(p_articles):
            p_articles = p_articles.sort_values( [ 'article_id',
                                                   'Keywords Similarity',
                                                   'User Articles Similarity', 
                                                   'User Articles Read',
                                                   'User Articles Interactions' ],
                                                 ascending = [True, False, False, False, False] )
            p_articles['Duplicate'] = p_articles.shift(1)['article_id'].fillna(-999)
            p_articles = p_articles[ p_articles['article_id'] != p_articles['Duplicate']]
            return p_articles.drop('Duplicate', axis = 1)
        
        self.__recommendations__ = pd.DataFrame()
        # We make the distinction between 3 cases:
        #     - Case 1: the given user has other similar users with at least 3 common articles
        #     - Case 2: the given user has read at least one article
        #     - Case 3: none of the above
        if max(self.__similarUsersID__['Articles Similarity']) > 3:            
            for idx in self.__similarUsersID__['Similar_Users'].index:
                v_similarUser = self.__similarUsersID__.loc[idx, 'Similar_Users']
                v_articles = self.getSimilarUsersDetails(v_similarUser).getUserArticles()                
                v_articles = setKeywordsSimilarity(v_articles)
                
                v_new_art = list(np.setdiff1d( v_articles['article_id'].tolist(), 
                                               self.__User__.getUserArticles()['article_id'].tolist(), 
                                               assume_unique = True ))   
                v_articles = v_articles.rename(columns = {'user_id': 'User ID'})
                v_articles = v_articles.set_index('article_id').loc[v_new_art, :].reset_index()
                v_articles['User Articles Similarity']   = self.__similarUsersID__.loc[idx, 'Articles Similarity']
                v_articles['User Articles Read']         = self.getSimilarUsersDetails(v_similarUser).getUserArticlesNo()
                v_articles['User Articles Interactions'] = self.getSimilarUsersDetails(v_similarUser).getUserArticlesInteract()
                v_articles = v_articles.sort_values( [ 'Keywords Similarity',
                                                       'User Articles Similarity', 
                                                       'User Articles Read',
                                                       'User Articles Interactions' ],
                                                     ascending = [False, False, False, False] )
                
                self.__recommendations__ = pd.concat([self.__recommendations__, v_articles.head(self.__artPerUser__)])
                self.__recommendations__.reset_index(drop = True, inplace = True)
                if self.__recommendations__.shape[0] > self.__recNumber__ * 4: 
                    break
            
            v_articles = self.__recommendations__.copy()
            v_articles['Article Similarity'] = np.NaN
            v_articles['Article Similar ID'] = np.NaN
            
            self.__recommendations__ = removeDuplicates(v_articles)
        
        elif max(self.getUser().getUserArticles()['article_id'].tolist()) > 0: 
            def getSimilarArticles(p_article_id, p_articles, p_articleSimilarity):
                '''
                INPUT
                    p_article_id - an article_id
                OUTPUT
                    v_similarArticles - an array of the most similar articles
                '''                
                v_article = pd.DataFrame(p_articleSimilarity.loc[p_article_id, :]).reset_index().astype(int)
                v_article.columns = ['article_id', 'Article Similarity']
                v_article = ( v_article[v_article['article_id'] != p_article_id]
                                       .sort_values('Article Similarity', ascending = False) )
                v_similarArticles = getArticles( p_articlesID = v_article['article_id'].tolist(), 
                                                 p_articles = p_articles ).merge(v_article, how = 'inner', on = 'article_id' )
                
                return v_similarArticles
            
            v_articles = pd.DataFrame()
            for article in self.__User__.getUserArticles()['article_id'].tolist():
                v_similarArticles = getSimilarArticles( p_article_id        = article, 
                                                        p_articles          = p_articles, 
                                                        p_articleSimilarity = p_articleSimilarity )
                v_similarArticles['Article Similar ID'] = article
                v_articles = pd.concat([v_articles, v_similarArticles])
            
            v_articles = v_articles.merge( self.__topArticles__[['article_id', 'Article Read No']], 
                                           how = 'inner', on = 'article_id' )
            
            v_articles['User ID']                    = np.NaN
            v_articles['User Articles Similarity']   = np.NaN
            v_articles['User Articles Read']         = np.NaN
            v_articles['User Articles Interactions'] = np.NaN
            
            self.__recommendations__ = removeDuplicates(setKeywordsSimilarity(v_articles))
                                
        else:
            v_articles = self.__topArticles__.copy()            
            v_articles['User ID']                    = np.NaN
            v_articles['User Articles Similarity']   = np.NaN
            v_articles['User Articles Read']         = np.NaN
            v_articles['User Articles Interactions'] = np.NaN
            v_articles['Article Similarity']         = np.NaN
            v_articles['Article Similar ID']         = np.NaN
            
            self.__recommendations__ = setKeywordsSimilarity(v_articles)
        
        v_cols = [ 'article_id', 'title', 'topic_keywords', 'Keywords Similarity', 
                   'Article Similarity', 'Article Similar ID', 'Article Read No', 
                   'User ID', 'User Articles Similarity', 'User Articles Read', 'User Articles Interactions' ]
        self.__recommendations__ = ( self.__recommendations__.sort_values( [ 'Keywords Similarity',
                                                                             'User Articles Similarity', 
                                                                             'User Articles Read', 
                                                                             'Article Similarity', 
                                                                             'Article Read No',
                                                                             'User Articles Interactions' ], 
                                                                           ascending = [False, False, False, False, False, False] )
                                                             .reset_index(drop = True)
                                                             .head(self.__recNumber__) )[v_cols]       
        return self.__recommendations__
            
    def __setSimilarUsers__(self, p_articles, p_userArticles, p_userArticlesMatrix, p_top):
        v_userID = self.__User__.getUserID()
        
        v_userArticlesMatrix = p_userArticlesMatrix.fillna(0)

        v_userArticleGroup = p_userArticles.groupby(['user_id', 'article_id']).agg({'user_id': ['count']}).reset_index()
        v_userArticleGroup.columns = ['user_id', 'article_id', 'Article Read No']
        v_userArticleGroup = v_userArticleGroup.groupby(['user_id']).agg({'Article Read No': ['count', 'sum']})
        v_userArticleGroup.columns = ['Articles Read', 'Articles Interactions']
        
        v_similarUsers = pd.DataFrame(v_userArticlesMatrix.dot(v_userArticlesMatrix.T).loc[v_userID, :])
        v_similarUsers.drop(v_userID, axis = 0, inplace = True)
        v_similarUsers.reset_index(inplace = True)
        v_similarUsers.columns = ['Similar_Users', 'Articles Similarity']

        v_similarUsers = v_similarUsers.merge( v_userArticleGroup.reset_index(), 
                                               how = 'inner', left_on = 'Similar_Users', right_on = 'user_id' )
        
        v_similarUsers = ( v_similarUsers.sort_values( ['Articles Similarity', 'Articles Read', 'Articles Interactions'], 
                                                       ascending = [False, False, False])
                                         .reset_index(drop = True)         
                                         .head(p_top) )    
        
        self.__similarUsersID__  = v_similarUsers[['Similar_Users', 'Articles Similarity']]
        self.__similarUsersDet__ = {}
        for idx in v_similarUsers.index:
            v_key = v_similarUsers.loc[idx, 'Similar_Users']
            self.__similarUsersDet__[v_key] = User( p_user_id            = v_key, 
                                                    p_articles           = p_articles, 
                                                    p_userArticles       = p_userArticles, 
                                                    p_userArticlesMatrix = p_userArticlesMatrix )
        
        return