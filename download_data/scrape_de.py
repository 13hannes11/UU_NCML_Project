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
WAIT_TIME_SEC = 7

# Filter
DATE_FROM = "17.10.2012" #dd.mm.yyyy
DATE_TO = "11.05.2021" #dd.mm.yyyy

# Output
DOWNLOAD_FOLDER = "../de/input/"


def get_element_by_xpath_or_false(driver, xpath):
    try:
        element = driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return element

def get_title_and_url():
    '''
    function to get titles and URLS for dataset
    '''
    title_url_list = []

    url = r'https://www.bundestag.de/parlament/plenum/abstimmung/liste'

    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=800,600")

    driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)
    driver.get(url)

    WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//table")))

    driver.find_element_by_xpath('//input[starts-with(@id, "from_")]').send_keys(DATE_FROM)
    driver.find_element_by_xpath('//input[starts-with(@id, "to_")]').send_keys(DATE_TO)
    driver.find_element_by_xpath('//div[@class= "bt-filterzeile-scroller"]').click()

    running = True
    while running:
        # as the side does not provide any loading indicators we need to wait after performing an action that requires loading
        print('waiting to be sure the page has updated')
        time.sleep(WAIT_TIME_SEC)

        # element selector to get elements that contain title and link to excel file
        element_selector = '//div[contains(@class, "bt-standard-conten") and not(@aria-hidden="true")]/table//div[@class= "bt-documents-description"]'
        elements = driver.find_elements_by_xpath(element_selector)

        for element in elements:
            if element.is_displayed():
                title_element = get_element_by_xpath_or_false(element, './p/strong')
                link_element = get_element_by_xpath_or_false(element, './ul/li/a[starts-with(@title, "XLS")]')
                if title_element and link_element:
                    title = title_element.text
                    link = link_element.get_attribute("href")
                    title_url_list.append((title, link))
                    print(title)
                    print(link)

        # Is there a next page
        element = get_element_by_xpath_or_false(driver, '//button[contains(@class, "slick-next") and not(contains(@class, "slick-disabled"))]')
        if element:
            # Move to bottom of page to avoid share button when clicking on element
            ActionChains(driver).move_to_element(driver.find_element_by_xpath('//div[contains(@class, "bt-footer-service")]')).perform()
            element.click()
        else:
            running = False
    driver.quit()
    return title_url_list


def save_to_file(file_url, folder):
    '''
    function to save file from url into specified folder
    '''

    file_name = file_url.split("/")[-1]
    req = requests.get(file_url)
    with open(folder + file_name,'wb') as output_file:
        output_file.write(req.content)
    return file_name

def save_titles(title_filename_list, folder):
    '''
    function to save title file mappings into folder as 'filename_to_titles.csv'
    '''
    with open(folder + 'filename_to_titles.csv', 'wt') as output_file:
        for title, file_name in title_filename_list:
            output_file.write(f'{file_name};{title}\n')

def title_url_list_element_saver(x:(str, str)):
    """ Function that downloads the file and returns a tuple of title and filename
    
    Args:
        x (str, str): tuple of title and url

    Returns:
        (str, str): returns a tuple of title and filename
    """
    print(f'Saving {x[0]}')
    return x[0], save_to_file(x[1], DOWNLOAD_FOLDER)


title_url_list = get_title_and_url()

pool = multiprocessing.Pool()
title_filenames_map = pool.map(title_url_list_element_saver, title_url_list)
save_titles(title_filenames_map, DOWNLOAD_FOLDER)

