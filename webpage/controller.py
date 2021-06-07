from flask import Flask, render_template

web = Flask(__name__)

@web.route("/")
@web.route("/home")
@web.route("/light")
def mainPage():
    return render_template("heropage-light.html")

@web.route("/dark")
def darkPage():
    return render_template("heropage-dark.html")


if __name__ == '__main__':
    web.run(debug= True)