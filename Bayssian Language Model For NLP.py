import numpy as np
import jieba
import time
import numpy as np


########################   Read text ##################################

path = 'data_set.txt'
f = open(path, 'r',encoding="utf-8")
input=f.read()
a=input.split()
f.close()
words_list=[]

wordsSp = jieba.cut(input)

for word in wordsSp:
    if (word !='\n')&(word!='')&(word!=' '):
        words_list.append(word)


#-----------------------------------------------------------------------   

book={}
book2={}
ibook={}
ibook2={}
for i in range(len(words_list)-2):
    word=words_list[i]
    nextword=words_list[i+1]
    nextnextword=words_list[i+2]
    
    ########################   Fill in books ##################################
    
    # 每個字下一個接的字的數量
    if word not in book:
        book[word] = dict()
        book[word][nextword] = 1
    else:
        if nextword not in book[word]:
            book[word][nextword] = 1
        else:
            book[word][nextword] += 1
    
    # 每個字下兩個接的字的數量
    if word not in book2:
        book2[word] = dict()
        book2[word][nextnextword] = 1
    else:
        if nextnextword not in book2[word]:
            book2[word][nextnextword] = 1
        else:
            book2[word][nextnextword] +=1
            
    # 每個字上一個接的字的數量
    if nextword not in ibook:
        ibook[nextword] = dict()
        ibook[nextword][word] = 1
    else:
        if word not in ibook[nextword]:
            ibook[nextword][word] = 1
        else:
            ibook[nextword][word] += 1
    
    # 每個字上兩個接的字的數量
    if nextnextword not in ibook2:
        ibook2[nextnextword] = dict()
        ibook2[nextnextword][word] = 1
    else:
        if word not in ibook2[nextnextword]:
            ibook2[nextnextword][word] = 1
        else:
            ibook2[nextnextword][word] +=1
    
    #-----------------------------------------------------------------------  
    

## two words
      
start='同學'
ans=start
ptr=start

stop='。'
stopCount=3

for i in range(100000):#   Generate 100000 !
    
    # 產生start下一個字
    if i <1: 
        val=list(book[ptr].values())  # "我"這個字下一個接的字的數量
        X=np.cumsum(val)  # 將接的字的數量做累加(array([0,0+1,0+1+2,...])
        pick=np.random.randint(X[0],X[-1]+1, size=10)[0]  # 從上面的array中取第一項與最後一項間任意整數，共10個，最後取第一個數
        choose=np.where(pick<=X)[0][0]
        ans+=list(book[ptr].keys())[choose]
        ptr=list(book[ptr].keys())[choose]
        
    else:
        a=list(book[ptr].keys())
        pa=[]
        pda1=[]
        pda2=[]
        for j in range(len(a)):
                
            ########################   Calcuate P(A)  ##############################
            
            pa = np.append(pa,np.sum(list(book[a[j]].values()))/len(words_list))  # 計算該字下兩個字的機率

            #---------------------------------------------------------------------
            
            ########################   Calcuate P(d1|A)  ###########################
                
            ikey = list(ibook[a[j]].keys())
            ival = list(ibook[a[j]].values())
            index = np.argwhere(np.array(ikey)==ptr)[0][0]  # 找尋ibook中，下個字的上個字為ptr的索引
            pda1 = np.append(pda1,ival[index]/np.sum(list(ival)))  # 計算ptr下個字的機率
            
            #-------------------------------------------------------------------------
            
            ########################   Calcuate Calcuate P(d2|A)   ##################
                
            ikey2 = list(ibook2[a[j]].keys())
            ival2 = list(ibook2[a[j]].values())
            try:
                index2 = np.argwhere(np.array(ikey2)==start)[0][0]
                pda2 = np.append(pda2,ival2[index2]/np.sum(list(ival2)))
            except:
                pda2 = np.append(pda2,0)
            
                
            #----------------------------------------------------------------------
        
            pa=np.array(pa)
            pda1=np.array(pda1)
            pda2=np.array(pda2)
            pad=pa*pda1*pda2
        
        ################  choose the action from p(a|d) distribution ################

        index_pda = np.argmax(pad)
        ans  += a[index_pda]
        ptr = a[index_pda]

        #-------------------------------------------------------------------------

    if stopCount==0:
        break
    else:
        stopCount=stopCount-1
        
        
    
print(ans)







