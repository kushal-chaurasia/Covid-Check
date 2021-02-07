from flask import Flask, render_template
import pickle 
app = Flask(__name__)

#file=open('model.pkl','rb')
#clf = pickle.load(file)
#file.close()


@app.route('/')
def hello():
    inputFeature= [100,1,22,-1,1]
    infProb=clf.predict_proba([inputFeature])[0][1]
    return render_template('contact_us.html')

if __name__ == '__main__':
    app.run()
