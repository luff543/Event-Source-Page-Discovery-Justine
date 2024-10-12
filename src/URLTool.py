import re
from url_normalize import url_normalize

def IsUrlString(string): 
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url = re.findall(regex,string)
    urls = [x[0] for x in url]
    if(len(urls)==0):
        return False
    return True

def GetNormalizaUrl(hrefs):
    hrefs = [i+'/' if i[-1] !='/' else i for i in hrefs]
    hrefs = ['http'+i[5:] if i[0:5] == 'https' else i for i in hrefs]
    return hrefs