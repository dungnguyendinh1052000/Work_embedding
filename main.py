from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
def model():
    #glove2word2vec(glove_input_file="/home/dung/Work/Work_embedding/glove.twitter.27B.200d.txt", word2vec_output_file="/home/dung/Work/Work_embedding/glove.twitter.27B.200d.word2vec.txt")
    glove_model = KeyedVectors.load_word2vec_format("/home/dung/Work/Work_embedding/glove.twitter.27B.200d.word2vec.txt", binary=False)
    
    return glove_model
glove_model=model()
def vector_represent(path_file):
    dataset=open(path_file,"r")
    vector_meeting=[]
    for line in dataset.read().split('\n'):
        try:
            line_text=line.split("__")[2]
            word=line_text.split()
            line_vector=[]
            i=0
            for index in range(len(word)):
                try:
                    line_vector.append(glove_model[word[index]])
                except KeyError:
                    i+=1
            line_vector=np.array(line_vector).reshape((200,len(word)-i))
            vector_meeting.append([line_vector,line.split("__")[1]])
        except IndexError :
            print(i)
    path_file='/home/dung/Work/Work_embedding/data_array/'+path_file.split('/')[-1].split('.')[0]
    np.save(path_file,vector_meeting)
import os
name_file=os.listdir("/home/dung/Work/Work_embedding/Data_Txt")
for addres in name_file:
    
    vector_represent(os.path.join("/home/dung/Work/Work_embedding/Data_Txt",addres))  