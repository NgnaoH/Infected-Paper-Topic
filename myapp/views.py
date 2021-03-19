from django.shortcuts import render
from bs4 import BeautifulSoup
import urllib.request
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

model = tf.keras.models.load_model("topicPaperModel.h5")
tokenizer = pickle.load(open("tokenizer.pickle", "rb"))


def isConnect(url):
    try:
        r = requests.get(url)
    except Exception as e:
        return False
    return True

def getContent(url):
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    content = soup.findAll("p")
    if content != None:
        return " ".join([p.text for p in content])
    return ""


def dashboard(request):
    dic = {
        0: "the-gioi",
        1: "du-lich",
        2: "the-thao",
        3: "giao-duc",
        4: "giai-tri",
        5: "phap-luat",
        6: "khoa-hoc",
        7: "suc-khoe",
        8: "kinh-doanh",
        9: "oto-xe-may",
    }

    responseData = []
    urls = request.POST.get("url")

    if urls != None and urls.strip() != "":
        urls = urls.split("\n")
        a = []
        ak = ""
        index = 0

        maxparalell = 30
        urlsnew = []
        with ThreadPoolExecutor(max_workers=min(maxparalell, len(urls))) as executor:
            future_to_url = {}
            for url in urls:
                url = url.strip()
                if url != "" and url != None and isConnect(url):
                    future_to_url[executor.submit(getContent, url)] = url

            for future in as_completed(future_to_url):
                urlsnew.append(future_to_url[future])
                a.append(future.result())

        urls = urlsnew

        if len(urls) > 0:
            a = np.array(a).reshape(-1,)
            a = tokenizer.texts_to_sequences(a)
            a = pad_sequences(a, maxlen=2029)
            ak = model.predict(a)

            for ii in ak:
                t = [[[], []], ""]
                for i in range(len(ii)):
                    t[0][0][0:0] = [dic[i]]
                    t[0][1][0:0] = [ii[i]]
                t[1] = urls[index]
                responseData.append(t)
                index += 1

    context = {"check": urls != None and len(urls) > 0, "responseData": responseData, "urls": urls}
    return render(request, "base.html", context)