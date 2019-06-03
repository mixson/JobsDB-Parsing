from bs4 import BeautifulSoup
import urllib
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import sys, re, string, datetime, getopt
from os import path
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

import os
import openpyxl

import time


def parse_job_post(target_website):
    html = requests.get(target_website)
    soup = BeautifulSoup(html.content)

    # search company information
    primary_comprofile_dictionary = {"data-timestamp": "p", "ref-jobsdb": "p", "ref-employer": "p",
        "jobad-header-company": "h2", "primary-profile-detail": "div" ,
                                     "general-pos": "h1",
                                     "jobad-primary-details": "div",
                                     }
    for search_key, search_identifier in primary_comprofile_dictionary.items():
        search_result = soup.find_all(search_identifier, attrs={"class": search_key})
        if search_result:
            search_result_text = search_result[0].text
        else:
            search_result_text = ""
        primary_comprofile_dictionary[search_key] = search_result_text

    primary_comprofile_name_dictionary = {"Post Date": "data-timestamp", "Post Ref": "ref-jobsdb", "Employer Ref": "ref-employer",
                                            "Company Name": "jobad-header-company", "Company Profile": "primary-profile-detail",
                                          "Job Name" : "general-pos",
                                          "Job Description": "jobad-primary-details"}

    for search_key, search_item in primary_comprofile_name_dictionary.items():
        primary_comprofile_name_dictionary[search_key] = primary_comprofile_dictionary[search_item]


    # search meta data
    jobpost_meta_dictionary = {"primary-meta-box row meta-lv": "div", "primary-meta-box row meta-edu": "div", "meta-industry": "div", "meta-jobfunction": "div"\
                               , "meta-location": "div", "primary-meta-box row meta-salary": "div","meta-employmenttype" : "div" , "meta-others": "div"\
                                ,"meta-benefit": "div"}
    for search_key, search_identifier in jobpost_meta_dictionary.items():
        search_result = soup.find_all(search_identifier, attrs={"class": search_key})

        if search_result:
            jobpost_meta_dictionary[search_key] = search_result[0].text
        else:
            jobpost_meta_dictionary[search_key] = ""

    job_post_meta_name_dictionary = {"Carrer Level": "primary-meta-box row meta-lv", "Qualification": "primary-meta-box row meta-edu"\
                                    , "Industry": "meta-industry", "Job Function": "meta-jobfunction", "Location": "meta-location"\
                                    , "Salary": "primary-meta-box row meta-salary", "Employment Type": "meta-employmenttype"\
                                    , "Others": "meta-others", "Benefits": "meta-benefit"}

    for search_key, search_item in job_post_meta_name_dictionary.items():
        if jobpost_meta_dictionary[search_item]:
            search_word_location = jobpost_meta_dictionary[search_item].find(search_key)
            text_start_location = search_word_location + len(search_key) + 1
            job_post_meta_name_dictionary[search_key] = jobpost_meta_dictionary[search_item][text_start_location :]


#search application email
    result = re.findall(r'([\w\.-]+@[\w-]+\.[\w\.-]*[\.]*[\w])', str(html.text))
    if result:
        email_dictionary = {"Email": result[0]}
    else:
        email_dictionary = {"Email": ""}
#search company website hyperlink
    company_website_dictionary = {"Company Website": ""}
    search_result = soup.find_all("img", attrs={"class": "jobad-header-logo"})
    if search_result:
        search_result_parent = search_result[0].parent
        if search_result_parent.name == "a":
            company_website_dictionary = {"Company Website": search_result_parent['href']}


    post_summary = {**primary_comprofile_name_dictionary, **job_post_meta_name_dictionary, **email_dictionary, **company_website_dictionary}
    return post_summary

if __name__ == "__main__":
    start_time = time.time()
    if (len(sys.argv) < 2):
        print(
            "\n\tUsage: jobsdbminer.py <search keywords> [-p <optional: search page count>] [-j <optional: HK junk boat style>]")
        print('\te.g. $pythonw jobsdbminer.py "HR Manager"')
        print('\te.g. $pythonw jobsdbminer.py "Web Developer" -p 5 -j\n')
        sys.exit(1)

    try:
        opts, args = getopt.getopt(sys.argv[2:], "jp:")
    except getopt.GetoptError as err:
        print('ERROR: {}'.format(err))
        sys.exit(1)

    junk_boat_mode = False
    search_page = 1

    for opt, arg in opts:
        if opt == '-j':
            junk_boat_mode = True
        elif opt == '-p':
            search_page = int(arg)

    params = sys.argv[1]
    params_search_word = params.replace(' ', '-').lower()

    url_prefix = "https://hk.jobsdb.com"
    params = "/hk/search-jobs/{}/".format(params_search_word)
    location = "Hong Kong"


    def getJobLinksFromIndexPage(soup, url_prefix):
        job_links_arr = []
        for link in tqdm(soup.find_all('a', attrs={'href': re.compile("^" + url_prefix + "/hk/en/job/")})):
            job_title_link = link.get('href')
            job_links_arr.append(job_title_link)
        return job_links_arr


    def getJobInfoLinks(url, next_page_count, url_prefix,
                        params, start_page):
        job_links_arr = []
        while True:
            # define an user agent as it is a required field for browsing JobsDB
            req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
            html = urllib.request.urlopen(req)
            soup = BeautifulSoup(html, 'lxml')
            job_links_arr += getJobLinksFromIndexPage(soup, url_prefix)

            start_page += 1
            print("Parsing Page {0}, {1} s".format(start_page, int(time.time() - start_time)))
            if (start_page > next_page_count):
                break
            next_page_tag = "{}{}".format(params, start_page)
            next_link = soup.find('a', attrs={'href': re.compile("^" + next_page_tag)})
            if (next_link == None):
                break
            url = url_prefix + next_link.get('href')
        return job_links_arr


    start_page = 1
    url = "{}{}{}".format(url_prefix, params, start_page)

    current_datetime = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print("Getting job links in {} page(s)...".format(search_page))
    job_links_arr = getJobInfoLinks(url, search_page, url_prefix,
                                    params, start_page)

    try:
        nltk.data.find('tokenizers/punkt')  # if nltk is not initialized, go download it
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

    punctuation = string.punctuation
    job_desc_arr = {}
    number_of_page = len(job_links_arr)
    print("Getting job details in {} post(s)...".format(number_of_page))
    for index, job_link in enumerate(job_links_arr):
        print("Parsing {0} -- {1} / {2}, {3}s".format(job_link, index, number_of_page, int(time.time()-start_time )))
        job_desc_arr[job_link] = parse_job_post(job_link)


    class excel_writer():
        def __init__(self, outputList):
            self.outputList = outputList

        def writeExcelRow(self, activeSheet, targetRow, ItemList):
            print("Writing Data to {} Row".format(targetRow))
            for columnNumber in range(1, len(ItemList) + 1):
                activeSheet.cell(row=targetRow, column=columnNumber).value = ItemList[columnNumber - 1]
            return activeSheet

        def writeExcelFile(self, filenName, activeSheet, columnNameList, outputList2Data, outputDirectory=os.getcwd()):
            wb = openpyxl.Workbook()
            sheet = wb.active
            startingRow = 1

            sheet = self.writeExcelRow(sheet, startingRow, columnNameList)
            for outputRow in outputList2Data:
                startingRow += 1
                sheet = self.writeExcelRow(sheet, startingRow, outputRow)
            outputFilePath = outputDirectory + "\\" + filenName
            wb.save(outputFilePath)


    # Write the result into excel file
    excel_writer = excel_writer(job_desc_arr)
    filename = params_search_word + ".xlsx"
    jobsdb_ref_webiste_dictionary = {}
    for keys, values in job_desc_arr.items():
        jobsdb_ref_webiste_dictionary[keys] = {"JobsDB Website": keys}
    for keys, values in jobsdb_ref_webiste_dictionary.items():
        job_desc_arr[keys].update(values)
    first_dictionary_item = next(iter(job_desc_arr.values()))
    column_name_list = [dictionary_key for dictionary_key, _ in first_dictionary_item.items()]
    outputList2Data = []
    for website, rowitems in job_desc_arr.items():
        # for name, row in rowitems.items():
            outputList2Data.append(list(rowitems.values()))
    excel_writer.writeExcelFile(filename, "", column_name_list, outputList2Data)