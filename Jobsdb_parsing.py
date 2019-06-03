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
params = params.replace(' ', '-').lower()

url_prefix = "https://hk.jobsdb.com"
params = "/hk/search-jobs/{}/".format(params)
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
job_desc_arr = []
print("Getting job details in {} post(s)...".format(len(job_links_arr)))
for job_link in tqdm(job_links_arr):
    req = urllib.request.Request(job_link, headers={'User-Agent': "Magic Browser"})
    job_html = urllib.request.urlopen(req)
    job_soup = BeautifulSoup(job_html, 'lxml')
    job_desc = job_soup.find('div', {'class': 'jobad-primary-details'})
    for li_tag in job_desc.findAll('li'):
        li_tag.insert(0, " ")  # add space before an object
    job_desc = job_desc.get_text()
    job_desc = re.sub('https?:\/\/.*[\r\n]*', '', job_desc, flags=re.MULTILINE)
    job_desc = job_desc.translate(job_desc.maketrans(punctuation, ' ' * len(punctuation)))
    job_desc_arr.append(job_desc)

stop_words = stopwords.words('english')
extra_stop_words = ["experience", "position", "work", "please", "click", "must", "may", "required", "preferred",
                    "type", "including", "strong", "ability", "needs", "apply", "skills", "requirements", "company",
                    "knowledge", "job", "responsibilities", "good", "related", "advantage", "salary", "candidates",
                    "expected", "interested", "working", "candidate", "used",
                    location.lower()] + location.lower().split()
stop_words += extra_stop_words
print("Generating Word Cloud...")
tfidf_para = {
    "stop_words": stop_words,
    "analyzer": 'word',  # analyzer in 'word' or 'character'
    "token_pattern": r'\w{1,}',  # match any word with 1 and unlimited length
    "sublinear_tf": False,
# False for smaller data size  #Apply sublinear tf scaling, to reduce the range of tf with 1 + log(tf)
    "dtype": np.float32,  # return data type
    "norm": 'l2',  # apply l2 normalization
    "smooth_idf": False,  # no need to one to document frequencies to avoid zero divisions
    "ngram_range": (1, 2),  # the min and max size of tokenized terms
    "max_features": 500  # the top 500 weighted features
}
tfidf_vect = TfidfVectorizer(**tfidf_para)
transformed_job_desc = tfidf_vect.fit_transform(job_desc_arr)

# Generate word cloud
freqs_dict = dict([(word, transformed_job_desc.getcol(idx).sum()) for word, idx in tfidf_vect.vocabulary_.items()])
plt.figure(figsize=(12, 9))
plt.title("Keywords:[{}] Location:[{}] {}".format(params, location, current_datetime))
if (junk_boat_mode):
    junkboat_mask = np.array(Image.open("images/junk_hk.png"))
    w = WordCloud(width=800, height=600, background_color='white', max_words=500, mask=junkboat_mask, contour_width=2,
                  contour_color='#E50110').fit_words(freqs_dict)
    image_colors = ImageColorGenerator(junkboat_mask)
    plt.imshow(w.recolor(color_func=image_colors), interpolation="bilinear")
else:
    w = WordCloud(width=800, height=600, mode='RGBA', background_color='white', max_words=500).fit_words(freqs_dict)
    plt.imshow(w)
plt.axis("off")
plt.show()
sorted_freqs_dict = sorted(freqs_dict.items(), key=lambda d: d[1], reverse=True)
