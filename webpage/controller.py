from flask import Flask, render_template
import GeneratePlots

web = Flask(__name__)

@web.route("/")
@web.route("/home")
@web.route("/light")
def mainPage():
    return render_template("heropage-light.html")

@web.route("/dark")
def darkPage():
    return render_template("heropage-dark.html")

@web.route("/regenerate/<theme>")
def regenerate(theme):
    if theme == 'light':
        back = '#fffffa'
        front = '#4a4a4a'
        GeneratePlots.generate(back, front, theme)
        return render_template("heropage-light.html")
    else:
        back = '#121f3b'
        front = '#fffffa'
        GeneratePlots.generate(back, front, theme)
        return render_template("heropage-dark.html")

if __name__ == '__main__':
    web.run(debug= True)