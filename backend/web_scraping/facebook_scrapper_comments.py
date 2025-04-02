import os
import json
import sys 
from sys import exit 
import numpy as np

import time
import urllib.request
import ssl
from dotenv import load_dotenv
from unicodedata import name
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from urllib.parse import urlparse, parse_qs

### Esta función es necesaria para evitar que el bot sea detectado como spam
def random_sleep(min_time, max_time):
    time.sleep(np.random.randint(min_time, max_time))


def login_facebook():
    load_dotenv()
    FACEBOOK_USER = os.getenv("facebook_user")
    FACEBOOK_PASSWORD = os.getenv("facebook_password")
    ## Loading the service and options for the chrome driver
    service = Service('E:/Users/1167486/Local/Drivers_web/chromedriver-win64/chromedriver')
    options = webdriver.ChromeOptions()
    options.binary_location = r'E:/Users/1167486/Local/Drivers_web/chrome-win64/chrome.exe'
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()
    driver.get('https://www.facebook.com/login/')
    random_sleep(3,7)
    ssl._create_default_https_context = ssl._create_unverified_context
    textos = []
    random_sleep(5,8)
    username = driver.find_element("css selector", "input[name='email']")
    password = driver.find_element("css selector", "input[name='pass']")
    username.clear()
    password.clear()
    random_sleep(5, 11)
    username.send_keys(FACEBOOK_USER)
    password.send_keys(FACEBOOK_PASSWORD)
    login = driver.find_element("css selector", "button[type='submit']").click()
    random_sleep(6, 11)
    

