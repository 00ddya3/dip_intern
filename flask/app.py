#플라스크 돌리는 코드
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

#선언, 인스턴스 설정
app = Flask(__name__)

#머신러닝 모형 불러오기
model=pickle.load(open('model2.pkl','rb'))

#딥러닝 모형 불러오기
tf_model = load_model("model/first_titanic.h5") #저장된 모델 로드
print('here')

#기본 라우트 설정(html)
@app.route('/')
def hello_world():
    return render_template("first_titanic.html")

#예측 라우터 설정, html에서 받아온 값을 입력값을 넣어주고 모형으로 예측
@app.route('/predict', methods=['POST','GET'])
def predict():
    #전처리
    int_features=[int(x) for x in request.form.values()]    #post를 통해 받아온 값을 저장
    final=np.array([int_features])
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)      #output 반환

#분류예측에 대한 결과값을 글로 표기
    if output>str(0.5):
        return render_template('first_titanic.html', pred='You may die.\nProbability of survive is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('first_titanic.html', pred='Your may survive.\n Probability of survive is {}'.format(output),bhai="Your Forest is Safe for now")

#예측 라우터 설정, html에서 받아온 값을 입력값을 넣어주고 모형으로 예측
@app.route('/predict2', methods=['POST','GET'])
def tf_predict():
    int_features=[int(x) for x in request.form.values()]
    final=np.array([int_features])
    output=tf_model.predict(final)

#분류예측에 대한 결과값을 글로 표기
    if output>0.5:
        return render_template('first_titanic.html', pred='You may die.\nProbability of survive is {}'.format(output), bhai="kuch karna hain iska ab?")
    else:
        return render_template('first_titanic.html', pred='Your may survive.\n Probability of survive is {}'.format(output), bhai="Your Forest is Safe for now")

#파이썬 파일을 바로 실행할때 사용
if __name__ == '__main__':
    app.run(debug=True)
