'''  
    1. 학습 map 생성 -> 교재로 학습
    2. 베이즈 추론 -> trainSet 으로 분류기 강화
    3. 강화된 map 을 기준으로 test set 분류
'''

#FileUtil class
import csv
class FileUtil:

    fileName = ''
    file = object()

    def __init__(self, fileName, option): # 생성자
        self.fileName = fileName
        self.file = open(self.fileName,mode=option, encoding='UTF-8', newline='')


    def csvFileRead(self): # csv 파일 읽기
        result = csv.reader(self.file)
        return result
    
    def csvFileWrite(self):
        result = csv.writer(self.file)
        return result

    def close(self): # Close File Ojbect 
        self.file.close()



#한국어 자연어 처리 클래스
import jpype
from konlpy.tag import Kkma 
from konlpy.utils import pprint


##TEST

"""파일 읽/쓰 세팅"""
trainFile = FileUtil("./hashcode_classification2020_train.csv", 'r')
trainData = trainFile.csvFileRead()

testFile2 = FileUtil("./hashcode_classification2020_test.csv", 'r')
testData2 = testFile2.csvFileRead()

resultFile = FileUtil("./hashcode_classification2020_result.csv", 'w')
wr = resultFile.csvFileWrite()

"""tf-idf, cosine-similarity 사용하여 분류 학습하고 test 유사도 측정해서 분류하기"""
"""추가적으로 사용 가능한 기술"""
"""LSA, 베이즈 추론"""

nlp = Kkma()

classification = {'1' : [], '2' : [], '3' :[], '4' : [], '5' : [], 'label' : []}

for d in trainData:
    for w in nlp.nouns(d[0]):
        if len(w) >= 3:
            classification[d[2]].append(w)
    for w in nlp.nouns(d[1]):
        if len(w) >= 3:
            classification[d[2]].append(w)
    for w in nlp.pos(d[0]):
        if w[1] == 'OL':
            classification[d[2]].append(w[0])
    try:
        for w in nlp.pos(d[1]):
            if w[1] == 'OL':
                classification[d[2]].append(w[0])
    except UnicodeDecodeError as e:
        continue
    
trainFile.close()


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

root = [' '.join(classification['1'])
            , ' '.join(classification['2'])
            , ' '.join(classification['3'])
            , ' '.join(classification['4'])
            , ' '.join(classification['5'])]

temp = []

for d in testData2:
    temp.append(' '.join(d))

root.extend(temp)

tfidfv = TfidfVectorizer(max_features= 1000,smooth_idf=True).fit(root)
category = tfidfv.transform(root).toarray()
#print(tfidfv.transform(root).toarray())
#print(tfidfv.vocabulary_)

#코사인 유사도 측정
from numpy import dot
from numpy.linalg import norm
import numpy as np

def cosineSimilarity(A, B):
       return dot(A, B)/(norm(A)*norm(B))







#docs = tfidfv.fit_transform(temp).toarray()

#print(docs)
for d in range(5, len(category)):
    max = 0.0
    for c in range(0, 5):
        sim = cosineSimilarity(np.array(category[d]),np.array(category[c]))
        if sim > max:
            max = sim
            classify = c+1

    wr.writerow([classify])


testFile2.close()
resultFile.close()

print('end')