from flask import Flask, render_template

web = Flask(__name__)

@web.route("/")
@web.route("/home")
def mainPage():
    return render_template("heropage.html")


if __name__ == '__main__':
    web.run(debug= True)