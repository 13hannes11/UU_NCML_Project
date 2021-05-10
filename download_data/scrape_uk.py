from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

import multiprocessing

import requests

import time

# Settings
DRIVER_PATH = "./chromedriver"
WAIT_TIME_SEC = 3

# Filter
DATE_FROM = "01/05/2020"
DATE_TO = "01/05/2021"

# Output
DOWNLOAD_FOLDER = "../uk/csv/"


def get_element_by_xpath_or_false(driver, xpath):
    try:
        element = driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return element

title_link_list = []
title_url_list = []
title_csv_list = []
#Function to get title url's
def get_all_link_urls():
   

    url = r'https://votes.parliament.uk/Votes/Commons'

    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=800,600")

    driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)
    driver.get(url)

    WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, '//div[contains(@class, "card-list")]')))

    driver.find_element_by_xpath('//*[@id="FromDate"]').clear()
    driver.find_element_by_xpath('//*[@id="ToDate"]').clear()
    driver.find_element_by_xpath('//*[@id="FromDate"]').send_keys(DATE_FROM)
    driver.find_element_by_xpath('//*[@id="ToDate"]').send_keys(DATE_TO)

    driver.find_element_by_xpath('//button[@class="btn btn-primary"]').click()

    running = True
    while running:
        # as the site does not provide any loading indicators we need to wait after performing an action that requires loading
        time.sleep(WAIT_TIME_SEC)
        
        # element selector to get elements that contain title and link to excel fil
        elem_selector = '//a[@class="card card-vote"]'
        elems = driver.find_elements_by_xpath(elem_selector)
    
        for elem in elems:
            if elem.is_displayed():
                title_url_list.append((elem.get_attribute("href")))
                print(f'Link to vote page: { elem.get_attribute("href") }') 

        # Is there a next page
        
        #last page
        elem_x = get_element_by_xpath_or_false(driver, '//li[@class="next"]') 
        if elem_x:
           elem_x.click()
        else:
            running = False
    driver.quit()
    return title_url_list   


   
#Function to get CSV file download url's
def get_all_file_links():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=800,600")

    driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)
    for elm in title_url_list:
        driver.get(elm)
        element = get_element_by_xpath_or_false(driver,'//a[2][@class="dropdown-item"]')
        element_x = element.get_attribute("href")
        print(f'Download url: {element_x}')
        title_link_list.append((elm, element_x))
        title_csv_list.append(element_x)
    driver.quit()
    return title_link_list


def save_to_file(file_url, folder):
    '''
    function to save file from url into specified folder
    '''

    file_name = file_url.split("/")[-1] + '.csv'
    req = requests.get(file_url)
    with open(folder + file_name,'wb') as output_file:
        output_file.write(req.content)
    return file_name

title_url_list = get_all_link_urls()
title_link_list = get_all_file_links()
for elem in title_link_list:
    print(elem)

for file_url in title_csv_list:
    print(f'saving: {file_url}')
    save_to_file(file_url, DOWNLOAD_FOLDER)
    
