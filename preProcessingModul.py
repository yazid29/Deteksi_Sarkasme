# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:40:05 2021

@author: AhmadYazidMunif
"""
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def lower(text):
    # lowercase
    lower = text.lower()
    return lower

def removeURLemoji(text):
    # hapus hastag/mention
    HastagRT = re.sub(r"#(\w+)|@(\w+)|(\brt\b)", " ", text)
    # hapus URL
    pola_url = r'http\S+'
    CleanURL = re.sub(pola_url, " ", HastagRT)
    # hapus emoticon
    hps_emoji = hapus_emoticon(CleanURL)
    # hapus multiWhitespace++, ex: ahh   haa
    text = re.sub('\s+', ' ', hps_emoji)
    # hasil akhir casefolding
    hasil = text
    return hasil

def angkadua(teksAwal2):
    final2 = []
    huruf2 = ""
    for x in range(len(teksAwal2)):
        cek2 = [i for i in teksAwal2[x]]
        for x in range(len(cek2)):
            if x == 0:
                final2.append(cek2[0])
                huruf2 = cek2[0]
            else:
                if cek2[x] != huruf2:
                    if cek2[x] == "2":
                        if(len(final2)) == 2:
                            final2.append(cek2[x-2])
                            final2.append(cek2[x-1])
                            huruf2 = cek2[x]
                        elif(len(final2) > 2):
                            jo = "".join(cek2[:2])
                            if(jo == "se" or jo == "di"):
                                final2.append(" ")
                                final2 = final2+cek2[2:x]
                                huruf2 = cek2[x]
                            else:
                                final2.append(" ")
                                final2 = final2+cek2[:x]
                                huruf2 = cek2[x]
                        else:
                            final2.append(cek2[x])
                            huruf2 = cek2[x]
                    else:
                        final2.append(cek2[x])
                        huruf2 = cek2[x]
                else:
                    final2.append(cek2[x])
                    huruf2 = cek2[x]
        final2.append(" ")
    hasil = "".join(final2).split()
    return hasil


def hapus_hurufganda(teksAwal):
    jml = 0

    final = []
    huruf = ""
    for x in range(len(teksAwal)):
        cek = [i for i in teksAwal[x]]
        for x in range(len(cek)):
            if x == 0:
                final.append(cek[0])
                huruf = cek[0]
                jml = 1
            else:
                if cek[x] != huruf:
                    final.append(cek[x])
                    huruf = cek[x]
                    jml = 1
                else:
                    if jml < 2:
                        final.append(cek[x])
                        huruf = cek[x]
                        jml += 1
        final.append(" ")
    hasil = "".join(final).split()
    return hasil


def hapus_simbolAngka(text):
    del_angkadua = angkadua(text)
    del_hrfganda = hapus_hurufganda(del_angkadua)

    # hasil=[]
    token = del_hrfganda
    lte = ["2g", "3g", "4g", "5g"]
    for i in range(len(token)):
        if(token[i] not in lte):
            token[i] = re.sub(r"\d+", " ", token[i])

    for ele in range(len(token)):
        token[ele] = token[ele].translate(
            str.maketrans(' ', ' ', string.punctuation))
        token[ele] = re.sub('\W', "", token[ele])
        token[ele] = re.sub('\s+', "", token[ele])

    return token


def hapus_simbolAngka2(text):
    token = text
    for i in range(len(token)):
        cekG = re.match(r"([\b234]+g)", token[i])
        if (cekG) == None:
            token[i] = re.sub(r"\d+", "", token[i])
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in token:
        if ele in punc:
            token = token.replace(ele, " ")
            token = re.sub('\s+', ' ', token)
    return token


def hapus_emoticon(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    # hapus emoji
    CleanEmoji = re.sub(emoji_pattern, "", text)
    return CleanEmoji


def tokenize(kalimat):
    return word_tokenize(kalimat)


def listokalimat(kalimat):
    listToStr = ' '.join(kalimat)
    return listToStr

"""
def delstopwordID(teks):
    notsinglechar=[]
    for kata in teks:
        a = re.sub(r"\b[a-zA-Z]\b", " ", kata)
        if(a!=" "):
            notsinglechar.append(a)
    return [kata for kata in notsinglechar if kata not in list_stopwords]

def daftarStopword():
    list_stopwords = stopwords.words('indonesian')
    # baca tambahan
    my_file = open("_stopwordTambahan.txt", "r")
    tambahan = my_file.read()
    daftar = tambahan.replace('\n', ' ').split()
    ####
    list_stopwords.extend(daftar)
    list_stopwords = set(list_stopwords)
    return list_stopwords
"""
def delstopword(teks,tambah):
    list_stopwords = stopwords.words('indonesian')
    if(len(tambah)==0):
        lst=daftartambah()
    else:
        dct = dict((x, y) for x, y in tambah)
        lst = list(dct.values())
    list_stopwords.extend(lst)
    list_stopwords = set(list_stopwords)
    
    notsinglechar=[]
    for kata in teks:
        a = re.sub(r"\b[a-zA-Z]\b", " ", kata)
        if(a!=" "):
            notsinglechar.append(a)
    return [kata for kata in notsinglechar if kata not in list_stopwords]

def daftartambah():
    # baca tambahan
    my_file = open("static/files/stopwordTambahan.txt", "r")
    tambahan = my_file.read()
    tambah = tambahan.replace('\n', ' ').split()
    return tambah

def normal_term():
    normalisasi_word = pd.read_excel("static/files/normalisasi.xlsx")
    normalisasi_dict = {}
    for index, row in normalisasi_word.iterrows():
        if row[0] not in normalisasi_dict:
            normalisasi_dict[row[0]] = row[1]
    return normalisasi_dict


def normalisasi(document):
    kalimat = document
    for term in range(len(kalimat)):
        if kalimat[term] in normalisasi_dict:
            kalimat[term] = normalisasi_dict[kalimat[term]]
    hasil = " ".join(kalimat).split()
    return hasil


def stemming(kalimat):
    term_dict = {}
    for kata in kalimat:
        for term in kalimat:
            if term not in term_dict:
                term_dict[term] = " "
    temp = list(term_dict)
    for x in range(len(temp)):
        if temp[x] == "jaringan":
            term_dict[temp[x]] = temp[x]
        elif temp[x] == "teh" and temp[x+1] == "anget":
            term_dict[temp[x]] = temp[x]
        else:
            term_dict[temp[x]] = stemmer.stem(temp[x])
    kalimat = [term_dict[term] for term in kalimat]
    #listToStr = ' '.join([str(i) for i in kalimat])
    return kalimat

def bulatkan(inte):
    new=round(inte,4)
    return new


term_dict = {}
factory = StemmerFactory()
stemmer = factory.create_stemmer()
normalisasi_dict = normal_term()
