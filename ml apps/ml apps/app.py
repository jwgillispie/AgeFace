from flask import Flask, render_template, request, redirect, flash
app = Flask(__name__)
# from inference import get_age
from inference import get_age

@app.route('/', methods = ["GET", 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")  
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files["file"]
        image = file.read()
        category = get_age(image_bytes=image)
      
        # predicted_age = "12"
        return render_template("result.html", age = category)


if __name__ =='__main__':
    app.run(debug=True)
