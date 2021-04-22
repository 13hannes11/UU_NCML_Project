from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains


import time

def get_element_by_xpath_or_false(driver, xpath):
    try:
        element = driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return element

WAIT_TIME_SEC = 7

DATE_FROM = "01.01.2019"
DATE_TO = "20.04.2021"

DRIVER_PATH = "./chromedriver"
url = r'https://www.bundestag.de/parlament/plenum/abstimmung/liste'

options = Options()
#options.add_argument("--headless")
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
    time.sleep(WAIT_TIME_SEC)

    # element selector to get elements that contain title and link to excel file
    element_selector = '//div[contains(@class, "bt-standard-conten") and not(@aria-hidden="true")]/table//div[@class= "bt-documents-description"]'
    elements = driver.find_elements_by_xpath(element_selector)

    for element in elements:
        if element.is_displayed():
            title_element = get_element_by_xpath_or_false(element, './p/strong')
            link_element = get_element_by_xpath_or_false(element, './ul/li/a[starts-with(@title, "XLSX")]')
            
            if title_element and link_element:
                title = title_element.text
                link = link_element.get_attribute("href")

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

# TODO: file downloader
