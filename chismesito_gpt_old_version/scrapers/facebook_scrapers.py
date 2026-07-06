import sys 
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import re 
import random

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium_stealth import stealth 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import time
import pandas as pd

from config import times 

def generateDeepFBSearch(query, 
                         facebook_email, 
                         facebook_password,
                         pages=1, 
                         wait_time=10):
    """
    Genera una búsqueda profunda en Facebook usando Selenium y la buena sopa bonita XD
    """
    FACEBOOK_LOGIN_URL = 'https://www.facebook.com/login/'
    CHROME_DRIVER_PATH = os.getenv("CHROME_DRIVER_PATH", 'E:/Users/1167486/Local/Drivers_web/chromedriver-win64/chromedriver.exe')
    CHROME_BINARY_PATH = os.getenv("CHROME_BINARY_PATH", r'E:/Users/1167486/Local/Drivers_web/chrome-win64/chrome.exe')

    #USERNAME_IN_COMMENT_XPATH ="span[class='x193iq5w xeuugli x13faqbe x1vvkbs x1xmvt09 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x4zkp8e x676frb x1nxh6w3 x1sibtaa x1s688f xzsf02u']"
    USERNAME_IN_COMMENT_XPATH2 = "span[class='x193iq5w xeuugli x13faqbe x1vvkbs x1xmvt09 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x4zkp8e x676frb x1nxh6w3 x1sibtaa x1s688f xzsf02u']"
    TIMESTAMP_IN_COMMENT_XPATH ="a[class='oajrlxb2 g5ia77u1 qu0x051f esr5mh6w e9989ue4 r7d6kgcz rq0escxv nhd2j8a9 nc684nl6 p7hjln8o kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x jb3vyjys rz4wbd8a qt6c0cv9 a8nywdso i1ao9s8h esuyzwwr f1sip0of lzcic4wl gmql0nx0 gpro0wi8 b1v8xokw']"
    COMMENT_TEXT_IN_COMMENT_XPATH = ".//div[@class='xdj266r x11i5rnm xat24cr x1mh8g0r x1vvkbs'] | .//div[@dir='auto' and not(.//a[@role='link'])]"
    #COMMENT_CONTAINER_SELECTOR = "div[class='xdj266r x11i5rnm xat24cr x1mh8g0r x1vvkbs']" # From your script, likely unstable
    COMMENT_CONTAINER_SELECTOR2="div[class='xdj266r x14z9mp xat24cr x1lziwak x1vvkbs']"
    TIMESTAMP_IN_COMMENT_XPATH2 = "a[class='x1i10hfl xjbqb8w x1ejq31n x18oe1m7 x1sy0etr xstzfhl x972fbf x10w94by x1qhh985 x14e42zd x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz xkrqix3 x1sur9pj xi81zsa x1s688f']"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

   # Configuración del driver de Selenium
    #### Comenzando con la implementación de Selenium para scraping de Facebook y semiautomatizar 
    def start_driver(driver_path, binary_path):
        print('Iniciando el driver de Chrome...')
        service = Service(driver_path)
        options = webdriver.ChromeOptions()
        options.binary_location = binary_path
        ### Añadimos elementos para mejorar la estabilidad
        options.add_argument("--disable-notifications")
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--lang-en-US')  
        #options.add_argument('--lang-es-MX')  # Ayuda con la consistencia del idioma
        options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Suprimir mensajes de DevTools
        ### Comenzar con el driver de Chrome
        driver = webdriver.Chrome(service=service, options=options)
        print('Driver de Chrome iniciado.')
        return driver


    def simulate_human_typing(element, text):
        """Simular como si un humano escribiera XD"""
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.1, 0.3))
            if random.random() < 0.1:
                time.sleep(random.uniform(0.3, 0.7))



    def login_to_facebook_get_cookies(driver,  email, password, login_url, wait_time=10):
        EMAIL_SELECTOR_LOGIN = "input[name='email']"
        PASSWORD_SELECTOR_LOGIN = "input[name='pass']"
        LOGIN_BUTTON_SELECTOR = "button[type='submit']"
        
        """Intentando loggearme a FB"""
        driver.maximize_window()
        driver.get(login_url)
        print('Intentando hacer login en FB')
        wait = WebDriverWait(driver, wait_time)

        # Ingresando el email
        email_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, EMAIL_SELECTOR_LOGIN))
        )
        simulate_human_typing(email_input, email)
        # Ingresando la contraseña
        
        password_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, PASSWORD_SELECTOR_LOGIN)) 
        )
        simulate_human_typing(password_input, password)
        print('Usuario y contraseña ingresados')
        login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, LOGIN_BUTTON_SELECTOR)))
        login_button.click()
        # Wait for login to complete - replace with a more reliable wait for a specific element on the home page
        print("Waiting for login to process...")
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, "//a[@aria-label='Home'] | //input[@aria-label='Buscar']")))
            print("Login exitoso.")
            return driver.get_cookies()
        except TimeoutException:
            print("Timed out waiting for a known element after login. Login might have failed or page structure changed.")
            time.sleep(5) # Fallback sleep
            #return None 
            return driver.get_cookies()

    def format_cookies_for_header(cookies_list):
        if not cookies_list: 
            return  ''
        return "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies_list])


    def extract_post_comments_to_dataframe_improved(
        driver: webdriver.Chrome,
        wait: WebDriverWait,
        post_url: str,
        container_selector: str,
        username_xpath: str,
        timestamp_xpath: str,
        comment_text_xpath: str,
        max_scroll_attempts: int = 10, # Renamed for clarity with dynamic scroll
        scroll_delay: float = 2.5
    ) -> pd.DataFrame:
        """
        Navigates to a Facebook post, scrolls dynamically to load comments, extracts
        username, timestamp, and comment text using provided selectors/xpaths,
        and returns them as a Pandas DataFrame.

        Uses dynamic scrolling to load as many comments as possible within
        max_scroll_attempts.

        Args:
            driver: The Selenium WebDriver instance.
            wait: The Selenium WebDriverWait instance.
            post_url: The URL of the Facebook post.
            container_selector: CSS selector for the main container of a single comment.
            username_xpath: XPath relative to the container to find the username.
            timestamp_xpath: XPath relative to the container to find the timestamp.
            comment_text_xpath: XPath relative to the container to find the comment text.
            max_scroll_attempts: Maximum number of times to attempt scrolling down.
                                Scrolling stops early if no new content is loaded.
            scroll_delay: Delay in seconds between scrolls.

        Returns:
            A Pandas DataFrame with columns: 'username', 'timestamp', 'comment', 'post_url'.
            Returns an empty DataFrame if the post or comments cannot be loaded.
        """
        logger.info(f"Navigating to post: {post_url}")
        driver.get(post_url)

        # Wait for a general post indicator.
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, "//div[@role='article'] | //div[contains(@aria-label, 'post')]")))
            logger.info("Post page appears to be loading.")
        except TimeoutException:
            logger.error("Timed out waiting for the main post content. Page structure might have changed or load is slow.")
            return pd.DataFrame(columns=['username', 'timestamp', 'comment', 'post_url']) # Return empty DataFrame

        # Dynamic scrolling to load comments
        logger.info(f"Scrolling down dynamically (max {max_scroll_attempts} attempts) to load comments...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        attempts = 0
        while attempts < max_scroll_attempts:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            logger.info(f"Scroll attempt {attempts + 1}/{max_scroll_attempts}")
            time.sleep(scroll_delay) # Wait for comments to load

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                logger.info("No new content loaded after scrolling. Stopping scroll.")
                break
            last_height = new_height
            attempts += 1
        logger.info("Finished scrolling attempts.")

        # Extracción de comentarios
        extracted_comments_data = []
        logger.info(f"Attempting to find comment containers using selector: {container_selector}")
        try:
            # Wait for at least one comment container to be present after scrolling
            # Use a shorter wait here as scrolling should have loaded them
            comment_containers = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, container_selector))
            )
            logger.info(f"Found {len(comment_containers)} potential comment containers.")
        except TimeoutException:
            logger.warning(f"Timed out waiting for comment containers with selector: {container_selector}")
            logger.warning("This could be due to: no comments, outdated selector, or comments not loading.")
            return pd.DataFrame(columns=['username', 'timestamp', 'comment', 'post_url'])
        except NoSuchElementException: # Should be caught by TimeoutException with presence_of_all_elements_located
            logger.warning(f"No comment containers found with selector: {container_selector}.")
            return pd.DataFrame(columns=['username', 'timestamp', 'comment', 'post_url'])
        except Exception as e:
            logger.error(f"An unexpected error occurred while finding comment containers: {e}")
            return pd.DataFrame(columns=['username', 'timestamp', 'comment', 'post_url'])


        logger.info("\n--- Extracting Comment Details ---")
        for i, container in enumerate(comment_containers):
            username = "N/A"
            timestamp = "N/A"
            comment_text = "N/A"
            container_text_preview = container.text[:100].replace('\n', ' ').strip() if container.text else 'EMPTY'

            try:
                # Find username within the container
                # Use find_elements here and check the list to avoid NoSuchElementException
                username_elements = container.find_elements(By.XPATH, username_xpath)
                if username_elements:
                    username_element = username_elements[0]
                    username = username_element.text.strip()
                    if not username: # Sometimes text is empty but in an attribute
                        username = username_element.get_attribute("aria-label") or username_element.get_attribute("textContent")
                        if username: username = username.strip()
                # else: logger.debug(f" - Username element not found for comment {i+1}") # Too noisy

            except StaleElementReferenceException:
                logger.warning(f" - Stale element reference for username in comment {i+1}. Skipping extraction for this element.")
            except Exception as e_user:
                logger.warning(f" - Error extracting username for comment {i+1} (Container preview: '{container_text_preview}'): {e_user}")


            try:
                # Find timestamp within the container
                timestamp_elements = container.find_elements(By.XPATH, timestamp_xpath)
                if timestamp_elements:
                    timestamp_element = timestamp_elements[0]
                    timestamp = timestamp_element.text.strip()
                    if not timestamp: # Timestamps often in 'aria-label' or 'title' of a link
                        timestamp = timestamp_element.get_attribute("aria-label") or timestamp_element.get_attribute("title") or timestamp_element.get_attribute("textContent")
                        if timestamp: timestamp = timestamp.strip()
                # else: logger.debug(f" - Timestamp element not found for comment {i+1}") # Too noisy

            except StaleElementReferenceException:
                logger.warning(f" - Stale element reference for timestamp in comment {i+1}. Skipping extraction for this element.")
            except Exception as e_time:
                logger.warning(f" - Error extracting timestamp for comment {i+1} (Container preview: '{container_text_preview}'): {e_time}")

            try:
                # Find comment text within the container
                comment_text_elements = container.find_elements(By.XPATH, comment_text_xpath)
                if comment_text_elements:
                    comment_text = comment_text_elements[0].text.strip()
                # else: logger.debug(f" - Comment text element not found for comment {i+1}") # Too noisy

            except StaleElementReferenceException:
                logger.warning(f" - Stale element reference for comment text in comment {i+1}. Skipping extraction for this element.")
            except Exception as e_text:
                logger.warning(f" - Error extracting comment text for comment {i+1} (Container preview: '{container_text_preview}'): {e_text}")

            # Check if any data was extracted for this container
            if username == "N/A" and timestamp == "N/A" and comment_text == "N/A":
                # If the container was found but no sub-elements, it might be an ad,
                # a "View more comments" link, or the sub-selectors are wrong.
                logger.debug(f"Comment {i+1}: Could not extract any specific details. Container preview: '{container_text_preview}'")
                continue # Skip if nothing useful was extracted

            extracted_comments_data.append({
                'username': username,
                'timestamp': timestamp,
                'comment': comment_text,
                'post_url': post_url
            })
            # logger.info(f"Comment {i+1}: User: '{username}', Time: '{timestamp}', Text: '{comment_text[:50]}...'") # Can be noisy

        logger.info("--- Finished Extracting Comment Details ---")
        df = pd.DataFrame(extracted_comments_data)
        logger.info(f"Extracted {len(df)} comments.")
        return df
    
    url = f'https://www.facebook.com/search/pages?q={query}'
    print(f"Starting search for '{query}' on Facebook.")

    driver = start_driver(CHROME_DRIVER_PATH, CHROME_BINARY_PATH)
    selenium_cookies = login_to_facebook_get_cookies(driver, facebook_email, facebook_password, 
                                                     FACEBOOK_LOGIN_URL, wait_time=wait_time) 
    formatted_cookie_string = format_cookies_for_header(selenium_cookies)
    driver.get(url)
    wait = WebDriverWait(driver, wait_time)  
    print('Obteniendo los contenedores de la búsqueda, es decir las páginas relacionadas con la búsqueda...')
    containers = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'xjp7ctv')))  
    links = [container.find_element(By.TAG_NAME, 'a') for container in containers if container.find_elements(By.TAG_NAME, 'a')]
    print('Dando click al primer enlace de la búsqueda...')
    links[0].click()
    print('Obteniendo los enlaces de los posts...')
    posts = driver.find_elements(By.XPATH, '//a[@role="link"]')
    posts_links = [post.get_attribute('href') for post in posts if post.get_attribute('href')]
    print('Obteniendo enlaces filtrados de los posts ')

    #filtered_items = [item for item in posts_links if re.search(r'fbid', item) and not re.search(r'login_alerts', item)] 
    filtered_items = [item for item in posts_links if re.search(r'fbid', item)] 
    print(f'Imprintiendo los enlaces filtrados de los posts length: {len(filtered_items)}')   
    print(filtered_items)    

    logger.info(f"Found {len(filtered_items)} posts matching the search query '{query}'.")
    print('Comenzando a extraer los comentarios de los posts filtrados...')
    list_df=  []
    for item in filtered_items:
        time.sleep(random.choice(times))  # Valores azarosos para simular no ser maquinillas
        x = extract_post_comments_to_dataframe_improved(
            post_url=item,
            driver=driver,
            wait=wait,
            container_selector=COMMENT_CONTAINER_SELECTOR2,
            username_xpath=USERNAME_IN_COMMENT_XPATH2,  
            timestamp_xpath= TIMESTAMP_IN_COMMENT_XPATH2,
            comment_text_xpath=COMMENT_TEXT_IN_COMMENT_XPATH,
            max_scroll_attempts=5,
            scroll_delay=2.5
        )    
        list_df.append(x)
    fbresults = pd.concat(list_df, ignore_index=True)
    fbresults['query'] = query  # Añadir la consulta de búsqueda a los resultados
    logger.info(f"Total comments extracted: {(fbresults.shape[0])}")
    driver.quit()  # Cerrar el driver al terminar 
    return fbresults