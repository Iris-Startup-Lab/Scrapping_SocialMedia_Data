from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

### Functions for scrapper 

def sign_in_with_twitter_account_selenium(email_or_phone_or_username, verify_account, password):
    try:
        driver = webdriver.Chrome()  
        driver.get("https://twitter.com/i/flow/login")

        login_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@data-testid="loginButton"]'))
        )
        login_button.click()
        time.sleep(2)
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
        username_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//input[@autocomplete="username"]'))
        )
        username_input.send_keys(email_or_phone_or_username)

        # Clic en el botón "Siguiente"
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//div[@role="button" and contains(., "Siguiente")]'))
        )
        next_button.click()

        try:
            verify_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//input[@autocomplete="off"]'))
            )
            verify_input.send_keys(verify_account)
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//div[@role="button" and contains(., "Siguiente")]'))
            )
            next_button.click()

        except TimeoutException:
            print("No se requirió verificación adicional.")

        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//input[@autocomplete="current-password"]'))
        )
        password_input.send_keys(password)
        login_submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//div[@data-testid="LoginForm_Login_Button"]'))
        )
        login_submit_button.click()
        time.sleep(5)

    except Exception as e:
        print(f"Ocurrió un error durante el inicio de sesión: {e}")
    finally:
        pass