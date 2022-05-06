from model import myModel, X_part, map_back, y_part
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from io import BytesIO
import base64
import os
import sys
sys.path.append(os.getcwd()+"//image_prediction//weights//model_weights_vision_transformer")
from image_model import theModel as imageModel, inputShape as imageInputShape
from flask import flash, redirect
from PIL import Image
imageModel.load_weights("image_prediction/weights/model_weights_vision_transformer")

from flask import Flask, render_template, request


app = Flask(__name__, template_folder = 'templates')
app.secret_key = "TwistedEveryWayWhatAnswerCanIGive!12bjkfndf"#need to load this in future
X_part.sort()
X_dict = {}

for i in range(len(X_part)):#O(1) access time after O(N) cost
    X_dict[X_part[i]] = i

def get_important(labels, odds, num_get = 6):#get k most likely labels
    print(len(labels))
    print(len(odds))
    assert(len(labels) == len(odds))
    top_five = [0] * num_get
    top_five_labels = [0] * num_get
    for i in range(len(odds)):
        consider = odds[i]
        if (consider>min(top_five)):
            replace = top_five.index(min(top_five))
            top_five[replace] = consider
            top_five_labels[replace] = labels[i]
    return top_five_labels, np.array(top_five)*100


#X_part = [i for i in X_part]

@app.route("/pic")
def pic():
    return render_template("upload_image.html")

@app.route("/process_img", methods = ["GET","POST"])
def process_img():
    print(request.method)

    output_to_prediction = ["AK","BCC","BKL","DF","NV","SCC","UNK","VASC"]
    if (request.method == "POST"):
        print("in POST")
            #print(request.form)
        if ('imagefile' not in request.files):
            print("files below")
            print(request.files)
            print(request.form)
            flash("no file part")
            return "<p>no image file</p>"
        file = request.files['imagefile']
        if (file.filename == ''):
            flash("no image selected for uplaoding")
            return "<p>no image selected for uploading</p>"
        filename = file.filename#secure_filename(file.filename)
        file.save(os.path.join("user_files", "temp_user.jpg"))
        output = imageModel.predict(np.array(Image.open(os.getcwd()+"//user_files//temp_user.jpg").resize((imageInputShape[0], imageInputShape[1])).getdata()).reshape(1,imageInputShape[0], imageInputShape[1],3))
        consider = get_important(["AK","BCC","BKL","DF","NV","SCC","UNK","VASC"], output[0], 7)
        abb_to_name = {
           "MEL": "melanoma",
           "NV": "melanocytic nevus",
           "BCC": "basal cell carcinoma",
           "AK": "actinic keratosis",
           "BKL": "benign keratosis",
           "DF" : "dermatofibroma",
           "VASC": "vascular leison",
           "SCC":"cutaneous squamous cell carcinoma",
           "UNK":"none of the others"
        }
        labels = [i + ": " + abb_to_name[i] for i in consider[0]]
        img = BytesIO()
        figure( figsize=(12+3+3, 10))

        plt.bar(consider[0], consider[1], width = 0.8)
        plt.ylabel("Predicted Odds (%)")
        plt.xlabel("Ailment")
        plt.title("Most Likely Condition")
        plt.savefig(img, format = 'png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return render_template("display_obj.html", userInput = plot_url, abb = labels)
        #flash("upload successful")
        #return "<h2>Run same as former on new model (import and load this time)</h2>"
    else:
        return "<h2>method is GET</h2>"

    print("in process_img")
    return "<h2>Internal error</h2>"



@app.route("/")
def index():
    return render_template("display.html", boxes = X_part)

@app.route("/results", methods = ["POST", "GET"])
def res():
    if (request.method=="GET"):
        print("in res GET")
        print(request.form)
        print(request.form.get(X_part[0]))
        return "get results"
    if (request.method == "POST"):
        #print(request.form)
        data = request.form
    #cat, per =
        #return "post"
        print("in POST of res!")
        user_input = np.zeros((1,len(X_part)))
        """
        print(X_part[0])
        print(X_part[-1])
        for part in X_part:
            print(part)

        """
        for key in data.keys():
            user_input[0,X_dict[key]] = 1
            print(key)
        print("input shape is " + str(user_input.shape))
        output = myModel.predict(user_input)[0]
        consider = get_important(y_part, output)

        img = BytesIO()
        figure( figsize=(12+3+3, 10))
        plt.bar(consider[0], consider[1], width = 0.8)
        plt.title("Most Likely Ailments")
        plt.xlabel("Ailment")
        plt.ylabel("Predicted Odds (%)")
        plt.savefig(img, format = 'png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        #plt.show()



        return render_template("display_obj.html", userInput = plot_url, abb = None)
    #return render_template("results.html", categories = cat, percentages = per)
if (__name__ == "__main__"):
    app.run()

    
