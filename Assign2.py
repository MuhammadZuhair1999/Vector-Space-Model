import string
import numpy as np
from tkinter import messagebox
from nltk.stem import WordNetLemmatizer
import math
import re
import tkinter as tk


'''testing function for GUI'''
def test_function(entry):
	print("This is the entry:", entry)

''' intialization of variables'''
lemmatizer = WordNetLemmatizer()
stopwords = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as", "once", " for", "at", "am", "are", "has", "have","had", "up", "his", "her", "in", "on", "no", "we", "do"]

N = 56           #Total number of documents
HEIGHT = 500     #height of GUI window
WIDTH = 1000      #Width of GUI window

tf_query = {}    #dictionary of term frequency of query'''
mydict = {}      #dictionary of total documents''' 
tf = {}          #dictionary of term frequency of documents'''
weight = {}      #tf*idf values of documents
df = {}          #dictionary of document frequency of documents'''
idf = {}         #dictionary of inverse document frequency of documents'''

score = []       #cosine similarity list 
unique = []      #total vocab list containing unique tokens  

def read_files(i):
    with open('Trump Speechs\speech_' + str(i) + '.txt', 'r') as inputfile:
        inputfile.readline()
        fetched = inputfile.read()                      #title removed
        inputfile = pre_processing(fetched)        
    return inputfile

'''function to optimize computation on document'''
def pre_processing(temp):
    temp = temp.lower()                               #casefolding
    temp = re.sub("[^A-Za-z0-9]+"," ",temp)          #removing punctuations
    word_tokens = temp.split()                       #tokenization
    d = []
    result = []
    for i in word_tokens:
        d.append(lemmatizer.lemmatize(i))             #lemmatization
    for j in d:
        if j not in stopwords:                       #removing stopwords
            result.append(j)                         #loading documents
    return result

'''to create index for vector space model''' 
def create_index():
    for docid in range(0,56):  
        key = docid
        c= read_files(docid)                        #reading dataset files
        mydict[key] = c                             #saving the documents  in index 
        for i in c:
            if i not in unique:                    #treating each document as unique document
                unique.append(i)   
    unique.sort()                                  #sorting the unique tokens
    return

'''term feature selection'''
'''function to calculate term frequency of document'''
def term_freq(tokens):           
   for i in range(0,56):
       tf[i] = {}
       for term in tokens:
           tf[i][term] = 0                         #initializing with 0 to avoid garbage values
           
   for i in range(0,56):
       for term in mydict[i]:
           tf[i][term] += 1                    # calculating term frequency of documents
           
           
'''function to calculate inverse document frequency'''
def inverse_document_frequency(term):
    return math.log(term/N)
   
'''function to calculate document frequency'''    
def document_freq():   
    for i in unique:                       #initializing with 0 values
       df[i] = 0
     
    for i in range(0,56):
       for term in unique:
           if(tf[i][term]==0):
               continue
           else:
               df[term] += 1                       # calculating document frequency 
   
    for i in unique:
        idf[i] = inverse_document_frequency(df[i])          # calculating idf of all unique tokens using df values
       
'''function to calculate tf*idf weight '''
def tfidf():
    for x in range(0,56):
        weight[x]={}  
        for i in unique:
            weight[x][i] = 0                    #initializing with 0 value
            
    for x in range(0,56):
        for word in mydict[x]:
            weight[x][word]=tf[x][word]*idf[word]   # storing tf*idf score in weight

'''function to calculate cosine similarity between query and document'''
def cosine_similarity(a,b):
    similarity = np.dot(a, b) / (math.sqrt(np.dot(a, a)) * math.sqrt(np.dot(b, b)))   #cosine similarity formula
    return similarity

'''function to optimize computation on query'''
'''function to perform preprocessing and postprocessing on query'''
def query_processing(words):
    words = words.lower().split()                               #query tokenization, casefolding
    table = str.maketrans('', '', string.punctuation)           #remove punctuation
    words = [w.translate(table) for w in words]
    words = [word for word in words if word not in stopwords]   #removing stopwords
    
    query_word = []                                            #temporary list for query tokens
    for j in words:
        if j not in stopwords:                                  #removing stopwords
            query_word.append(lemmatizer.lemmatize(j))          #lemmatize query
   
    for i in unique:                      #initialize term freq of query list with 0 
        tf_query[i] = 0
       
    for i in query_word:                  #calculating term frequency of query           
        tf_query[i]+=1
    
    weight_query = {}                     #dictionary for storing tf*idf values for query
    
    for x in unique:
        weight_query[x]=0                 #initializing weight query dictionary key with 0 value
    
    for key in query_word:
        weight_query[key]=tf_query[key]*idf[key]       #calculating tf*idf values of query

    temp_list  = []                              #termporary list for vectorization 
    for i in range(0,56):
        temp_list = list(weight[i].values())         
        score.append(cosine_similarity(list(weight_query.values()),temp_list))   #saving cosine similrity values in score list         
    
    return score

'''function to display the output'''
def display(query):
    try:
        score = query_processing(query)
        alpha = 0.0005
        output = []
        print(" Query : ", query)
        for i in range(len(score)):
            if score[i] > alpha:                            #filtering documents using alpha value                           
                output.append(i)
                print(" Document# ",i," Similarity = ", score[i])
        print("Length = : ",len(output))                    #calculating length (no of docs after filteration )
        print(output)
        final_str = 'Documents Retrieved From VSM Retrieval Model \nDataset: %s \nQuery: %s \nAplha value: %s \nLength: %s \n %s' % ("Trump Speechs",query,alpha,len(output),output)
    except:
        final_str = 'There was a problem retrieving that information\n Invalid Query!'
    return final_str

def clicked():
    messagebox.showinfo('Notice:', 'Success! \nMy some results are 100% accurate at different alpha and rest on other alpha value (e.g: "muslim" length = 2 output = [3,4] at alpha  = 0.05) \nso kindly check on different alpha values \n\nFor next query  Run Assign2.py again to avoid result override')
    
def GUI_result(query):
    label['text'] = display(query)
    label1 = clicked()

    
'''calling functions'''    
create_index()
term_freq(unique)
document_freq()
tfidf()

#print("Search Query")
#x = input()

#query_processing(x)

''' GUI primitives'''
root = tk.Tk()
root.title("Vector Space Model")
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

background_image = tk.PhotoImage(file='landscape.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)


frame = tk.Frame(root, bg='#fa9484', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

entry = tk.Entry(frame, font=40)
entry.place(relwidth=0.65, relheight=1)

button = tk.Button(frame, text="Search Query",bg = "black", fg="yellow", font=40, command=lambda: GUI_result(entry.get()))
button.place(relx=0.7, relheight=1, relwidth=0.3)

lower_frame = tk.Frame(root, bg='#fa9484', bd=10)
lower_frame.place(relx=0.5, rely=0.25, relwidth=1, relheight=0.6, anchor='n')

label = tk.Label(lower_frame)

label.place(relwidth=1, relheight=1)

button2 = tk.Button(root, text="QUIT",bg = "black", fg="yellow",font = 100,command=root.destroy)
button2.pack(side="bottom")
       
root.mainloop()
