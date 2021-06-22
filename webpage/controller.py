from flask import Flask, render_template, request
import GeneratePlots
import df_stuff

web = Flask(__name__)
unis, type_, diff = df_stuff.fetch_list()

@web.route("/", methods= ["POST", "GET"])
@web.route("/home", methods= ["POST", "GET"])
@web.route("/light", methods= ["POST", "GET"])
def mainPage():
    if request.method == "POST":
        uniSelect = request.form['uniselect']
        typeSelect = request.form['typeselect']
        diffSelect = request.form['diffselect']
        GeneratePlots.get_summary(uniSelect, typeSelect, diffSelect, 'light')
        return render_template("heropage-light.html", universities= unis, type_= type_, difficulty= diff, goforit= True)
    return render_template("heropage-light.html", universities= unis, type_= type_, difficulty= diff, goforit= False)

@web.route("/dark", methods= ["POST", "GET"])
def darkPage():
    if request.method == "POST":
        uniSelect = request.form['uniselect']
        typeSelect = request.form['typeselect']
        diffSelect = request.form['diffselect']
        GeneratePlots.get_summary(uniSelect, typeSelect, diffSelect, 'dark')
        return render_template("heropage-dark.html", universities= unis, type_= type_, difficulty= diff, goforit= True)
    return render_template("heropage-dark.html", universities= unis, type_= type_, difficulty= diff, goforit= False)

@web.route("/regenerate/<theme>", methods= ["POST", "GET"])
def regenerate(theme):
    if theme == 'light':
        back = '#fffffa'
        front = '#4a4a4a'
        GeneratePlots.generate(back, front, theme)
        if request.method == "POST":
            uniSelect = request.form['uniselect']
            typeSelect = request.form['typeselect']
            diffSelect = request.form['diffselect']
            GeneratePlots.get_summary(uniSelect, typeSelect, diffSelect, theme)
            return render_template("heropage-light.html", universities= unis, type_= type_, difficulty= diff, goforit= True)
        return render_template("heropage-light.html", universities= unis, type_= type_, difficulty= diff, goforit= False)
    else:
        back = '#121f3b'
        front = '#fffffa'
        GeneratePlots.generate(back, front, theme)
        if request.method == "POST":
            uniSelect = request.form['uniselect']
            typeSelect = request.form['typeselect']
            diffSelect = request.form['diffselect']
            GeneratePlots.get_summary(uniSelect, typeSelect, diffSelect, theme)
            return render_template("heropage-dark.html", universities= unis, type_= type_, difficulty= diff, goforit= True)
        return render_template("heropage-dark.html", universities= unis, type_= type_, difficulty= diff, goforit= False)

if __name__ == '__main__':
    web.run(debug= False)
