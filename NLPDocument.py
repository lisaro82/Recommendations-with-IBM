import re
from string import punctuation

# Try to import the libraries for nltk. In case of error, download the necessary components and try again.
for _ in range(2):
    try:
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.stem.porter import PorterStemmer        

        g_StopWordsEnglish = stopwords.words('english')
        v_stopWords = [ 'new', 'like', 'u', "it'", "'s", "n't", 'mr.', 'from', 'subject', 're', 'edu', 'use', 'table' ]
        g_StopWordsEnglish.extend(v_stopWords)
        g_StopWordsEnglish = list(set(g_StopWordsEnglish))

        break
    except:
        import nltk
        nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'words'])

        
#-------------------------------------------------------------------------------------------------------------------------------------
class NLPDocument():
    
    __punctuation      = None
    __usePorterStemmer = None
    __filterMinSize    = None
    __document         = None
    __excludeTags      = None
    __includeTags      = None
    
    __clean_document   = None
    __clean_tokens     = None
    __pos_tags         = None
    
    def __init__( self, p_document, 
                  p_usePorterStemmer = True, 
                  p_filterMinSize    = None, 
                  p_excludeTags      = None, 
                  p_includeTags      = None ):
        self.__punctuation      = punctuation + "”“‘’»'"
        self.__document         = ( p_document.lower().replace("'ll ", ' will ')
                                                      .replace("'re ", ' are ')
                                                      .replace("'ve ", ' have ') )
        self.__usePorterStemmer = p_usePorterStemmer    
        self.__filterMinSize    = p_filterMinSize 
        self.__excludeTags      = p_excludeTags 
        self.__includeTags      = p_includeTags
        
        self.__wordTokenize__()        
        return
    
    def getTokenizedDoc(self):
        return self.__clean_document
    
    def getCollocations(self):
        return self.__clean_document.collocations()
    
    def getPosTag(self):
        return self.__pos_tags
    
    def printPosTag(self, p_top = 20):
        for key, value in self.__pos_tags.items():
            print(f'{key.ljust(6, " ")} has {len(value)} occurence(s). {value[:p_top]}')        
        return
    
    def __sentTokenize__(self):
        for item in ['¿', '--', '*']:
            self.__document = self.__document.replace(item, ' ')
            
        self.__document = self.__document.replace(' www.', 'http://')
                            
        # Remove URLs
        v_url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        v_urls = re.findall(v_url_regex, self.__document)
        for url in v_urls:
            self.__document = self.__document.replace(url, "urlplaceholder")

        # Remove Emails
        v_email_regex = '\S*@\S*\s?'
        v_emails = re.findall(v_email_regex, self.__document)
        for email in v_emails:
            self.__document = self.__document.replace(email, "emailplaceholder") 
        
        return nltk.sent_tokenize(self.__document)
    
    def __wordTokenize__(self):
        v_tokens = []
        for token_by_sent in [word_tokenize(sentence) for sentence in self.__sentTokenize__()]:
            v_tokens += token_by_sent
            
        v_tokens = list(filter(lambda item: item not in g_StopWordsEnglish, v_tokens))
        v_tokens = list(filter(lambda item: item not in punctuation, v_tokens))
        v_tokens = list(filter(lambda item: item not in [u"...", u'``', u'\u2014', u'\u2026', u'\u2013'], v_tokens))
                        
        self.__clean_tokens = []
        for token in v_tokens:
            token = WordNetLemmatizer().lemmatize(token)
            token = WordNetLemmatizer().lemmatize(token, pos = 'v')
            
            if not self.__filterMinSize is None:
                if len(token) < self.__filterMinSize:
                    continue
                
            if self.__usePorterStemmer:
                token = PorterStemmer().stem(token)
            
            self.__clean_tokens.append(token.strip())
        
        v_pos_tag = nltk.pos_tag(self.__clean_tokens)
                
        if not self.__includeTags is None:
            self.__clean_tokens = [item[0] for item in v_pos_tag if item[1] in self.__includeTags]
        
        if not self.__excludeTags is None:
            self.__clean_tokens = [item[0] for item in v_pos_tag if item[1] not in self.__excludeTags]
            
        self.__clean_document = ' '.join(self.__clean_tokens)
                
        self.__pos_tags = {}
        for item in nltk.pos_tag(self.__clean_tokens):
            v_key = item[1]
            if not v_key in self.__pos_tags.keys():
                self.__pos_tags[v_key] = []
            self.__pos_tags[v_key].append(item[0])
            
        self.__pos_tags = {key: sorted(set(value)) for key, value in self.__pos_tags.items()}
        
        return self.__clean_document