from bs4 import BeautifulSoup
from selenium import webdriver
import requests

"""
Call `scrap_data` from the file where you need data from
"""

user_agent_desktop = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'

headers = { 'User-Agent': user_agent_desktop}

data= {"university": [],
       "course": [],
       "type": [],
       "review": [],
       "votes": [],
       "studets": [],
       "difficulty": []}

driver = webdriver.Chrome('chromedriver') 
def move_page(number):
    url = f"https://www.coursera.org/search?query=python&page={number}&index=prod_all_products_term_optimization"
    driver.get(url) 
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    courses = soup.find_all("li", attrs= {"class": "ais-InfiniteHits-item"})
    return courses
import time
def scrap_data():
    for page in range(1, 97):
        print("Page: ", page, "Length: ", len(data["university"]))
        courses = move_page(page)
        time.sleep(3)
        for course in courses:
            tail = course.find("div", attrs= {"class": "rc-ProductInfo"})
            head = course.find("div", attrs= {"class": "card-content"})

            uni = head.find("span", attrs= {"class": "partner-name"})
            data['university'].append(uni.text if uni else None)

            course = head.find("h2", attrs= {"class": "color-primary-text"})
            data['course'].append(course.text if course else None)

            type_ = head.find("div", attrs= {"class":"_jen3vs"})
            data['type'].append(type_.text if type_ else None)

            # --

            review = tail.find("span", attrs= {"class":"ratings-text"})
            data['review'].append(review.text if review else None)

            votes = tail.find("span", attrs= {"class":"ratings-count"})
            data['votes'].append(votes.text if votes else None)

            students = tail.find("span", attrs= {"class":"enrollment-number"})
            data['studets'].append(students.text if students else None)

            difficulty = tail.find("span", attrs= {"class":"difficulty"})
            data['difficulty'].append(difficulty.text if difficulty else None)
    return data