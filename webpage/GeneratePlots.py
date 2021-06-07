import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set();

plt.rcParams['figure.figsize'] = 20, 10
plt.style.use('dark_background')
plt.rcParams['axes.grid'] = False

df = pd.read_pickle('../data/scraped_data')

from matplotlib import cm

def get_color_spectrum(start, end, n, flipped= False, spectrum = "seismic"):
    if flipped:
        return eval(f'cm.{spectrum}(np.linspace(start, end, n))[::-1]')
    return eval(f'cm.{spectrum}(np.linspace(start, end, n))')

def plot_bar_values(plot, values, xoffset= 1, yoffset= 0.2, type_ = 'v', fontdict= None, **kwargs):
    '''Parameters:
    --------------
    • plot: 
    x = plt.plot()
    Then `x` becomes the `plot`
    
    • values: series.values
    
    • xoffset, yoffset - Self explanatory
    
    • type_: This should be 'h' or 'v' depending on the type of your plot.
    '''
    for patch, val in zip(plot.patches, values):
        x = patch.get_width() if type_ == 'h' else patch.get_x()
        y = patch.get_y() if type_ == 'h' else patch.get_height()
        plt.text(x + xoffset, y + yoffset, str(val), fontdict= fontdict, **kwargs)

def generate(background= "#fffbf2", front= "#424242"):
    plt.rcParams['figure.facecolor'] = background
    plt.rcParams['font.family'] = 'Neuville'
    plt.rcParams['text.color'] = front
    plt.rcParams['axes.labelcolor'] = front
    plt.rcParams['axes.facecolor'] = background
    plt.rcParams['xtick.color'] = front
    plt.rcParams['ytick.color'] = front

    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False

    savepath = "./static/"

    print("[STARTING]")
    # 1
    fig = plt.figure(dpi= 250)
    ax = plt.axes()
    ax.set_facecolor(background)
    ax.set(xlabel= "Course Count", ylabel= "University")
    topUni = df.university.value_counts()[:10][::-1]
    plot = plt.barh(topUni.index, topUni.values, color= get_color_spectrum(.1, .9, 10, flipped= False, spectrum= 'Blues'))
    plot_bar_values(plot, topUni.values, type_= 'h', xoffset= 5)
    fig.savefig(savepath + 'top10uniCourses.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 1")


    # 2
    ax = plt.axes()
    ax.set_facecolor(background)
    topUni = df[df.type != 'GUIDED PROJECT'].university.value_counts()[:10][::-1]
    plot = plt.barh(topUni.index, topUni.values, color= get_color_spectrum(.4, .6, 10, flipped= False, spectrum= 'Blues'))
    plot_bar_values(plot, topUni.values, type_= 'h')
    ax.set(xlabel= "Course Count", ylabel= "University")
    desc = \
    '''
    So, if we try to look at the
    data witout the GUIDED PROJECTS
    IBM is the top programs provided
    in comparision to the rest'''

    ax.text(90, 2, desc, fontsize= 30, fontweight= 'light', ha= 'right');
    plt.savefig(savepath + 'top10uniCourses2.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 2")


    # 3
    fig = plt.figure()
    ax1 = plt.axes()
    courseType = df.type.value_counts()
    plot = plt.bar(courseType.index, courseType.values, color= get_color_spectrum(.8, .5, len(courseType), flipped= False, spectrum= 'Blues'))
    plot_bar_values(plot, courseType.values, xoffset= 0.35, yoffset= 5)
    ax1.set_yticks([])
    ax2 = fig.add_axes([0.4, 0.5, 0.3, 0.5])

    topUniDF = df[df.university.isin(topUni.index[:-1])]
    sns.countplot(y= 'university', data= topUniDF, hue= 'type', palette= ['r', 'g', 'b', 'y'], saturation= 1, ax= ax2)
    ax2.yaxis.tick_right()
    plt.legend(loc= [0.2, 0.65])
    ax2.set_xticks([])
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_xlim([60, 0])
    ax2.set_title("Top universities provides what")
    plt.savefig(savepath + 'coursetype.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 3")


    # 4
    uniRanked = df.groupby("university")[['review', 'votes']].agg({"review": "mean", "votes": "sum"}).sort_values(by= "votes", ascending= False)
    uniRanked.dropna(inplace= True)
    topRankers = uniRanked[:10]
    bottomRankers = uniRanked[-10:]
    both = pd.concat([topRankers, bottomRankers])[::-1]

    fig, ax = plt.subplots(1, 1, figsize= (20, 15), dpi= 200)
    ax.barh(both.index, 6, color= 'grey', alpha= 0.05)
    ax.barh(both.index, both.review)
    ax.set_yticklabels([])
    for side in ['right', 'top', 'bottom']:
        ax.spines[side].set_visible(False)
    for patch in ax.patches[20:25]:
        patch.set_color((1, 0, 0, 0.5))
    for patch in ax.patches[35:]:
        patch.set_color((0, 1, 0, 0.5))
    for patch in ax.patches[25:35]:
        patch.set_color((.71, .71, .71,0.2))
        
    for patch, val, names in zip(ax.patches[20:], both.votes, both.index):
        x = patch.get_width()
        y = patch.get_y()
        ax.text(x + 0.1, y + 0.25, str(int(val)))
        ax.text(x - 0.1, y + 0.25, str(names), ha= 'right', fontfamily= "product sans")
        
    ax.set_xticks(range(0,6));
    ax.set_title("Average reviews of top / bottom 10 universities (Sorted by votes)")
    ax.set_xlabel("Average review out of 5")
    plt.savefig(savepath + 'reviews.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 4")


    # 5
    fig = plt.figure(dpi= 200)
    uniStud = df.groupby("university")['students'].sum().sort_values(ascending= False).round(1)
    unis = uniStud[:45]
    plt.bar(unis.index, unis.values, color= get_color_spectrum(.0, .8, len(unis), spectrum= "Blues", flipped= True));
    plt.xticks(rotation= 90);
    plt.hlines(uniStud.mean(), xmin= 0, xmax= uniStud.index[-1], ls= '--', lw= 2, color= front)
    plt.annotate("Mean: 518092 Students", xy= ("University of London", 518092), xytext= (30, 2018092),
                arrowprops= dict(arrowstyle= "->", connectionstyle= "arc3, rad= -0.2", color= front));

    title = 'Students enrollment'
    desc = \
    '''
    Here, it seems that University of Michigan is on the top at
    teaching programs about python. It is on the top by reviews
    and also by its students' count. Noticibly, it is on the second
    rank about its number of courses on the platform.
    '''
    plt.text(10, 90_00_000, title, fontdict= dict(fontsize= 40, fontweight= 'bold'))
    plt.text(10, 65_00_000, desc, fontdict= dict(fontsize= 20, fontweight= 'light'));
    plt.savefig(savepath + 'students.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 5")

    
    # 6
    fig = plt.figure()
    ax = fig.add_axes([1,1,1,1])
    ax2 = fig.add_axes([1.5,1.5,0.5,0.5])
    sns.countplot(x= "difficulty", data= df, hue= 'type', ax = ax, saturation= 1)

    diff = df.difficulty.value_counts()
    plot = ax2.bar(diff.index, diff.values, color= get_color_spectrum(.7, .9, 1, spectrum= "Blues"))
    ax2.set_yticks([])
    plot_bar_values(plot, diff.values, xoffset= 0.38, yoffset= -35, fontdict={"fontfamily": "product sans", "size": 20}, ha= 'center')
    plt.savefig(savepath + 'difficulty.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 6")


    # 7
    Beginner = df[df.difficulty == 'Beginner'].sort_values(by= "students", ascending= False)[:3]
    Intermediate = df[df.difficulty == 'Intermediate'].sort_values(by= "students", ascending= False)[:3]
    Mixed = df[df.difficulty == 'Mixed'].sort_values(by= "students", ascending= False)[:3].copy()
    Mixed.iloc[1, 1] = 'Programming for Everybody\n(Getting Started with Python)'
    Advanced = df[df.difficulty == 'Advanced'].sort_values(by= "students", ascending= False)[:3]
    
    import matplotlib.lines as line
    fig = plt.figure(figsize= (30, 10))
    ax = plt.axes()
    ax.set(xticks= [], yticks= [])
    for side in ["left", 'right', 'bottom', 'top']:
        ax.spines[side].set_visible(False)

    plt.text(0.45, 0.9, "Top 3 courses by difficulty level", ha= "center",
            fontfamily= 'Neuville', fontweight= "bold", fontsize= 50)

    l1 = line.Line2D([0.2, 0.8], [0.75, 0.75], transform=fig.transFigure, figure=fig, color = 'grey', linestyle='-',linewidth = 3, alpha = 0.6)
    fig.lines.extend([l1])

    for course, x, color in zip(["Beginner", "Intermediate", "Mixed", "Advanced"], [0.1, 0.35, 0.60, 0.85], get_color_spectrum(0, .4, 4, spectrum= 'rainbow', flipped= True)):
        plt.text(x, -0.1, course, ha= "center",  fontfamily= 'Neuville', fontweight= "bold", fontsize= 40, color= color)
        for row, ofset in zip(eval(course)[['course', 'university']].iterrows(), np.arange(0.2, 0.7, 0.2)):
            plt.text(x, ofset, row[1][0], ha= "center", fontfamily= 'Neuville', fontweight= 'regular', fontsize= 25, color= color)
            plt.text(x, ofset - 0.04, row[1][1], fontfamily= 'Neuville', fontweight= 5, fontsize= 15,  ha= "center", color= color)
            
    l2 = line.Line2D([0.28, 0.28], [0.1, 0.70], transform=fig.transFigure, figure=fig, color = 'grey', linestyle='-',linewidth = 3, alpha = 0.3)
    l3 = line.Line2D([0.50, 0.50], [0.1, 0.70], transform=fig.transFigure, figure=fig, color = 'grey', linestyle='-',linewidth = 3, alpha = 0.3)
    l4 = line.Line2D([0.68, 0.68], [0.1, 0.70], transform=fig.transFigure, figure=fig, color = 'grey', linestyle='-',linewidth = 3, alpha = 0.3)
    fig.lines.extend([l2, l3, l4])
    plt.savefig(savepath + 'difficultyCourses.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 7")


    # 8
    import re
    
    pattern = r'(\bA\.?I\.?\b)|(\bM\.?L\.?\b)|(\bD\.?L\.?\b)|(\bN\.?L\.?P\.?\b)(Artificial Intelligence)|(Machine Learning)|(Deep Learning)|(Reinforcement Learning)|(Tensor\s?Flow)|(Natural Language Processing)|(Neural Networks?)'
    with_MlAiDl = df[df.course.str.match(pattern, flags= re.IGNORECASE)]
    plt.figure(dpi= 200)
    ax = plt.axes()
    plot = with_MlAiDl.university.value_counts()[::-1].plot(kind= 'bar', color= get_color_spectrum(0.1, .7, 14, spectrum= 'Blues'))
    plot_bar_values(plot, with_MlAiDl.university.value_counts()[::-1].values, type_= 'v', xoffset= 0.2, yoffset= 0.1)
    ax.set_xlabel("Universities with corses on AI")
    ax.set_ylabel("Count")

    title = "Universities that provide programs on AI"
    desc = \
    '''
    To get the list of those universities, I had to do a little bit of "textual engineering".
    I extracted those courses with the relavent keywords like AI, ML, DL,
    Deep Learning, Tensorflow, and many more... and also with their combinations
    with capitals and smalls. Then after giving almost all keywords, I found the data
    which can be plotted like this.
    '''
    ax.set_yticks([])
    ax.text(0, 27, title, fontsize= 35)
    ax.text(0, 18, desc, fontsize= 18, fontweight= 'light');
    plt.savefig(savepath + 'AICourses.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 8")


    # 9
    top5_AI = with_MlAiDl.university.value_counts().index[:5]
    top5_AI_DF = with_MlAiDl[with_MlAiDl.university.isin(top5_AI)]
    top5_other_DF = df[df.university.isin(top5_AI) & ~(df.index.isin(top5_AI_DF.index))]
    AI_vs_REST = pd.DataFrame({"AI": top5_AI_DF.groupby("university")['students'].sum(), "REST": top5_other_DF.groupby("university")['students'].sum()})

    fig = plt.figure(figsize= (20, 20))
    ax1 = fig.add_axes([0, 0.5, 0.3, 0.3])
    ax2 = fig.add_axes([0.3, 0.5, 0.3, 0.3])
    ax3 = fig.add_axes([0.6, 0.5, 0.3, 0.3])

    ax4 = fig.add_axes([0.15, 0.2, 0.3, 0.3])
    ax5 = fig.add_axes([0.45, 0.2, 0.3, 0.3])

    for idx, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
        ax.set(xticks= [], yticks= [])
        patches, _, __ = ax.pie(AI_vs_REST.iloc[idx], colors= get_color_spectrum(.3, .35, 2, spectrum= 'jet'), autopct= "%.2f%%")
        ax.set_xlabel(AI_vs_REST.index[idx])
    ax2.legend(patches, ["AI", "REST"], loc=9);
    plt.savefig(savepath + 'AIpie.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 9")


    # 10
    import string
    from nltk.corpus import stopwords
    puncs = string.punctuation
    pattern = r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]'
    top50_terms = df.iloc[df.votes.sort_values(ascending= False).index][:50]
    
    def remove(str_):
        return re.sub(pattern, '', str_)
    
    top50_terms.course = top50_terms.course.apply(remove)
    terms = top50_terms.course.str.lower().str.get_dummies(" ")
    stopwords = stopwords.words("english")
    stopwords.pop(26)
    
    valids = terms.columns[~(terms.columns.str.lower().isin(stopwords))]
    terms = terms.loc[:, valids]
    most_used = terms.sum(axis= 0).sort_values(ascending= False)
    
    ax = plt.axes()
    _, _, autopects = ax.pie(most_used[:15], colors= get_color_spectrum(.2, .9, 15, spectrum= "Blues", flipped= True), labels=most_used.index.str.title()[:15], 
        autopct= "%.1f", pctdistance=.9)
    my_circle=plt.Circle( (0,0), 0.8, color=background)
    plt.setp(autopects, **{'color':'white', 'weight':'bold', 'fontsize':15.5})

    ax.add_artist(my_circle)
    plt.savefig(savepath + 'ring.png', bbox_inches= 'tight', transparent= True, dpi= 250)
    print("[DONE] 10")


    # 11
    from wordcloud import WordCloud, STOPWORDS
    courses = df.course.apply(remove)
    all_terms = []
    courses.apply(lambda x: all_terms.extend(x.lower().split()))
    all_terms = pd.Series(all_terms)
    all_terms = all_terms[~(all_terms.isin(stopwords))]
    all_terms.drop(all_terms[all_terms == "using"].index, inplace= True)
    
    text = ' '.join(all_terms)
    plt.rcParams['figure.figsize'] = (12,12)
    wordcloud = WordCloud(background_color = background, colormap='Blues', width = 1200,  height = 1080, max_words = 200).generate(text)
    
    print('DONE.')
    return True