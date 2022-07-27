#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

#해당 파일은 모형(pkl)을 만드는 코드임
#라이브러리 불러오기
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

#데이터 불러오기
data = pd.read_csv("titanic.csv")
data = data[['Sex', 'Age', 'Fare', 'Survived']]
data['Age'] = data['Age'].fillna(data.Age.mean())       #결측치 제거
data['Sex'] = data['Sex'].apply(lambda x : 0 if x=='female' else 1)     #라벨인코딩
#data = np.array(data)

#독립, 종속변수 분리, 타입변경
X = data.drop(columns=['Survived'])
y = data['Survived']
y = y.astype('int')
X = X.astype('int')

#훈련용, 테스트 분리, 모형선언
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()

#모형학습
log_reg.fit(X_train, y_train)

#45, 32, 60을 인풋값으로 만들기
#inputt=[int(x) for x in "45 32 60".split(' ')]
#final=[np.array(inputt)]

#예측도 해보기(테스트용)
#b = log_reg.predict_proba(final)

#log_reg 객체를 파일형태로 저장
pickle.dump(log_reg, open('model2.pkl','wb'))

#파일을 로드, 확인용
model=pickle.load(open('model2.pkl','rb'))


