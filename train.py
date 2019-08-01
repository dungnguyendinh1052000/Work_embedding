from keras.layers import LSTM,Dense,GlobalMaxPool1D,Dropout,TimeDistributed
from keras.models import Sequential
import numpy as np 
from sklearn.model_selection import train_test_split
import os

def ham(x,y):
    name_file=os.listdir("drive/My Drive/Data_Txt")
    n=[]
    k=[]
    for addres in name_file[x:y]:
      a=vector_represent(os.path.join("drive/My Drive/Data_Txt",addres))
    
      for i in range(len(a)):
        n.append(a[i][0].T)
        k.append(a[i][1].T)
    n=np.array(n).reshape((len(n),90,200))
    k=np.array(k).reshape((len(k),56))
    return n,k
# b=[]
# name_file=os.listdir("drive/My Drive/data_array_56label")
# for addres in name_file:
#     b.append(os.path.join("drive/My Drive/data_array_56label",addres))
# for i in range(50):
#     ham(b[i])
n,k=ham(0,50)

X_train, X_test, y_train, y_test=train_test_split(n,k,test_size=0.33,shuffle=True)
model=Sequential()
model.add(LSTM(100,input_shape=(90,200),return_sequences=False))
model.add(Dropout(0.5))
#model.add(TimeDistributed(Dense(100)))
#model.add(GlobalMaxPool1D())
model.add(Dense(56,activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(56,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=128)
X_test,y_test=ham(60,70)
score, acc = model.evaluate(X_test, y_test)
print(acc)
# import os
# max_=[]
# def max_len(a):
    
#     print(a[:][0].reshape((a.shape[0],200,90)).shape)
# name_file=os.listdir("/home/dung/Work/Work_embedding/data_array")
    
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
label=["s",'qy','qw','qr','qrr','qo','qh','fg','fh','h','b','bk','ba','bh','aa','aap','na','ar','arp','nd','ng','am','no','co','cs','cc','f','br','bu','r','m','bs','bc','bsc','df','e','2','bd','by','fa','ft','fw','x','z','%','%-','%--','fe','t','tc','j','t1','t3','d','g','rt']

def model_1():
    #glove2word2vec(glove_input_file="/home/dung/Work/Work_embedding/glove.twitter.27B.200d.txt", word2vec_output_file="/home/dung/Work/Work_embedding/glove.twitter.27B.200d.word2vec.txt")
    glove_model = KeyedVectors.load_word2vec_format("drive/My Drive/glove.bin", binary=True)

    
    return glove_model

def vector_input(input_):
    vector=[]
    glove_model=model_1()
    i=0
    for word in input_.split():
        try:
            vector.append(glove_model[word])
        except KeyError:
            i+=1
    vector=np.array(vector).reshape((200,len(input_.split())-i))
    vector=np.concatenate((vector,np.zeros((200,90-vector.shape[1]))),axis=1)  
    return vector
x=vector_input("uhhuh").reshape((1,90,200))
print(np.argmax(model.predict(x)))
