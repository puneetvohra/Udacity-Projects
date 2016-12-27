from selenium import webdriver
from selenium.webdriver.common.by import By
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import pandas as pd
import caps_common as cc
import config as cfg

def is_visible(driver, locator, timeout=2):
    try:
        ui.WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((By.ID, locator)))
    except:
        return False
    return True

def get_fundamental_data():
	'''
	Downloads fundamental data 
	'''
	tickers = cc.get_ticker_list()
	ticker_info = pd.DataFrame(index = tickers, columns = ['Name','Summary','Stock Type','Site','Market Cap','Net Income','Sales','Employees','Sector','Industry'])
	binary = FirefoxBinary(cfg.firefox_loc)
	driver = webdriver.Firefox(firefox_binary=binary)
	for ticker in tickers:

		driver.get('http://financials.morningstar.com/company-profile/c.action?t=' + ticker + '&region=USA&culture=en_US')

		if not is_visible(driver, "OperationDetails"):
			continue
	
		try:
			summary = driver.find_element_by_class_name('r_txt6').text
			tbl0 = driver.find_element_by_class_name('r_table0').text
			tbl1 = driver.find_element_by_class_name('r_table1')
			first_row = tbl1.find_elements_by_class_name('text3')[0].text
			second_row = tbl1.find_elements_by_class_name('text3')[1]
			second_row = second_row.find_elements_by_tag_name('td')
			contact_info = driver.find_element_by_id("ContactInfo").text
			title = driver.find_element_by_class_name('r_title').text
			details = driver.find_element_by_id("OperationDetails").text
			details = details.split('\n')
			ticker_info.loc[ticker,'Sector'] = str(second_row[2].text)
			ticker_info.loc[ticker,'Industry'] = str(second_row[4].text)
		except:
		
			continue

		try:
			ticker_info.loc[ticker,'Name'] = str(title.split('  ')[0])
		except:
			pass
		try:
			ticker_info.loc[ticker,'Summary'] = str(summary)
		except:
			pass
		try:
			ticker_info.loc[ticker,'Stock Type'] = str(tbl0.split('\n')[1].split(' ')[0])
		except:
			pass
		try:
			ticker_info.loc[ticker,'Site'] = str(contact_info.split(' ')[-1])
		except:
			pass
		try:
			ticker_info.loc[ticker,'Market Cap'] = str(first_row.split(' ')[1])
		except:
			pass
		try:
			ticker_info.loc[ticker,'Net Income'] = str(first_row.split(' ')[2])
		except:
			pass
		try:
			ticker_info.loc[ticker,'Sales'] = str(second_row[0].text)
		except:
			pass
		try:
			for row in details:
				if 'Employees' in row:
					employees = str(row.split(' ')[-1])
			ticker_info.loc[ticker,'Employees'] = employees
		except:
			pass

	driver.quit()

	ticker_info.dropna(thresh = 3).to_csv('data/fundamental_data.csv', encoding = 'utf-8')
	return
