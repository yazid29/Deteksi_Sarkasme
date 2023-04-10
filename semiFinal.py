# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:26:49 2021
https://hackersandslackers.com/flask-routes/
@author: AhmadYazidMunif
"""
import os
import os.path
import pandas as pd
import pymysql
import preProcessingModul
import tweepy
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from collections import Counter
from flask import Flask, flash, request, redirect, url_for, render_template, session, jsonify, request, json, send_from_directory
from datetime import datetime, timedelta
import pickle
from myServer_maybe import databaseServer
from flask_bcrypt import Bcrypt

# app setting
config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 10,
    "SECRET_KEY": "get-stronger"
}
project_root = os.path.dirname(os.path.realpath('__file__'))
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')
download_path = os.path.join(project_root, 'uploads')
myserver = databaseServer()
app = Flask(__name__, template_folder=template_path, static_folder=static_path)
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_TYPE'] = 'redis'
app.config['SECRET_KEY'] = 'get-stronger'
bcrypt = Bcrypt(app)
print(app.secret_key)


def connectDB():
    hoste = myserver.hoste
    usere = myserver.usere
    passworde = myserver.passworde
    dbe = myserver.dbe
    connection = pymysql.connect(
        host=hoste, user=usere, password=passworde, db=dbe)
    return connection


def selectDB(connection, tabel):
    cursor = connection.cursor()
    if (tabel == 'dataset'):
        query = "SELECT * FROM `dataset`"
    elif (tabel == 'dataset2'):
        query = "SELECT * FROM `dataset2`"
    elif (tabel == 'preprocessing'):
        query = "SELECT * FROM `preprocessing`"
    elif (tabel == 'preprocessing2'):
        query = "SELECT * FROM `preprocessing2`"
    elif (tabel == 'stopword'):
        query = "SELECT * FROM `stopword`"
    elif (tabel == 'tbl_user'):
        query = "SELECT * FROM `tbl_user`"
    cursor.execute(query)
    selectData = cursor.fetchall()
    return selectData, cursor


@app.route("/login", methods=['GET', 'POST'])
def login():
    connection = connectDB()
    cursor = connection.cursor()
    if request.method == 'POST':
        username = request.form['uname']
        passworde = request.form['passworde']
        query = "SELECT * FROM `tbl_user` WHERE `username`= %s"
        cursor.execute(query, (username,))
        selectData = cursor.fetchall()
        if (len(selectData) == 1):
            namA = selectData[0][1]
            passW = selectData[0][2]
            leveL = selectData[0][3]
            cek = bcrypt.check_password_hash(passW, passworde)
            if (cek == True):
                if (leveL == 1):
                    session['uname'] = 'admin'
                    session['nama'] = namA
                    session['level'] = '2'
                elif (leveL == 2):
                    session['uname'] = username
                    session['nama'] = namA
                    session['level'] = '2'
            else:
                flash("Username/Password Salah")
        else:
            flash("Username/Password Salah")
    return redirect("/")


@app.route("/login2", methods=['GET', 'POST'])
def login2():
    session['uname'] = "User"
    session['nama'] = "User"
    session['level'] = '3'
    return redirect("/")


@app.route("/updateData", methods=['POST'])
def updateData():
    connection = connectDB()
    cursor = connection.cursor()
    # datae
    id_data = request.form['id']
    username = request.form['uname']
    label = request.form['label']
    query = "UPDATE `dataset2` SET `%s` = '%s' WHERE `dataset2`.`No` = %d" % (
        username, label, int(id_data))
    print(query)
    # username,label,id_data
    cursor.execute(query)
    connection.commit()
    print(username, "dan", label)
    return jsonify({'result': 'success'})


@app.route("/out", methods=['GET', 'POST'])
def out():
    session['uname'] = None
    session['nama'] = None
    session['level'] = None
    return redirect("/")


@app.route("/", methods=['GET', 'POST'])
def index():
    connection = connectDB()
    selectData, cursor = selectDB(connection, "tbl_user")
    if (session.get("uname") == None):
        view = render_template('login.html')
    else:
        username = session.get("uname")
        level = session.get("level")
        pilihData = session.get("pilih")
        if (pilihData == "dataset" or pilihData == "dataset2"):
            selectData, cursor = selectDB(connection, pilihData)
            DataTweet = pd.DataFrame(selectData, columns=[
                                     "No", "Tweet", "Label"])
            jumlahData = DataTweet["Tweet"].shape[0]  # Jumlah Data
            jmlSarkas = DataTweet[DataTweet.Label == "Sarkasme"].shape[0]
            jmlNonSarkas = DataTweet[DataTweet.Label == "BukanSarkas"].shape[0]
            cekJumlah = pd.DataFrame(data={'Jumlah': [jumlahData],
                                           'Sarkasme': [jmlSarkas],
                                           'BukanSarkas': [jmlNonSarkas]},
                                     columns=['Jumlah', 'Sarkasme', 'BukanSarkas'])
            print('\n...... Jumlah Data ...... ')
            print(cekJumlah)
            data = {"pilihData": pilihData,
                    "preproses": session.get("preproses"),
                    "tfidf": session.get("tfidf"),
                    "train_model": session.get("train_model"),
                    "jumlahTotal": jumlahData,
                    "sarkasme": jmlSarkas,
                    "bukansarkas": jmlNonSarkas}
        else:
            data = {"pilihData": pilihData,
                    "preproses": session.get("preproses"),
                    "tfidf": session.get("tfidf"),
                    "train_model": session.get("train_model"),
                    "jumlahTotal": 0,
                    "sarkasme": 0,
                    "bukansarkas": 0}
        view = render_template('index.html', data=data)
    return view


@app.route("/selectDataset", methods=["POST"])
def selectDataset():
    connection = connectDB()
    if request.method == "POST":
        session["pilih"] = request.form['selectData']
        pilihData = session.get("pilih")
        if (pilihData == "dataset"):
            namaFile = "static/files/tweet_label_dikit.csv"
            query = "INSERT INTO `dataset` (`Tweet`, `Label`) VALUES (%s, %s)"
            selectData, cursor = selectDB(connection, pilihData)
            if (len(selectData) == 0):
                print("Data Belum Ada")
                DataTweet = pd.read_csv(namaFile)
                print('insert tweet awal')
                for row in DataTweet.values.tolist():
                    cursor.execute(query, (row[0], row[1]))
                connection.commit()
                print("Data Berhasil Ditambah")
                print(DataTweet)
        print(pilihData)
    return redirect("/")


@app.route("/gantiData")
def gantiData():
    session["pilih"] = None
    session["preproses"] = None
    session["tfidf"] = None
    session["train_model"] = None
    session["rata2akurasi"] = None
    session["kfold"] = None
    return redirect("/")


@app.route("/dataset")
def godataset():
    print("\nDataset")
    connection = connectDB()
    pilihData = session.get("pilih")
    print(pilihData)
    if (pilihData != None):
        # namaFile="static/files/#tweet_label_dikit.xlsx"
        #DataTweet = pd.read_excel(namaFile)
        selectData, cursor = selectDB(connection, pilihData)
        DataTweet = pd.DataFrame(selectData, columns=["No", "Tweet", "Label"])
        print(DataTweet.head(2))
        jmlahData = len(DataTweet)
        data = {"preproses": session.get("preproses"), "tfidf": session.get("tfidf"),
                "train_model": session.get("train_model")}
        view = render_template(
            'dataset.html', dataset=DataTweet, data=data, jmlahD=jmlahData)
    else:
        view = redirect("/")
    return view


@app.route("/stopword")
def view_stopword():
    print("\nStopword")
    connection = connectDB()
    stopword, cursor = selectDB(connection, 'stopword')
    if (len(stopword) == 0):
        tambahan = preProcessingModul.daftartambah()
        aStopword = pd.DataFrame(tambahan, columns=["word"])
        aStopword = aStopword.sort_values(by=["word"])
        # insert database
        query = "INSERT INTO `stopword` (`word`) VALUES (%s)"
        for row in aStopword["word"].values.tolist():
            cursor.execute(query, (row))
        connection.commit()
    else:
        aStopword = pd.DataFrame(stopword, columns=["stop_id", "word"])
        aStopword = aStopword.sort_values(by=["word"])
    stopwordlist = aStopword
    jmlStopword = len(stopwordlist)
    data = {"preproses": session.get("preproses"), "tfidf": session.get("tfidf"),
            "train_model": session.get("train_model")}
    return render_template('stopword.html', data=data, stopwordlist=stopwordlist, jmlStop=jmlStopword)


@app.route("/addStopword", methods=["POST"])
def addStopword():
    cursor = connection.cursor()
    if request.method == "POST":
        stpword = request.form['stopword']
        # insert database
        query = "INSERT INTO `stopword` (`word`) VALUES (%s)"
        cursor.execute(query, (stpword))
        connection.commit()
    return redirect("/stopword")


@app.route("/deleteStp/<string:id>")
def deleteStp(id):
    cursor = connection.cursor()
    cursor.execute("DELETE FROM `stopword` WHERE (`stop_id`) = (%s)", (id))
    connection.commit()
    return redirect("/stopword")


@app.route("/view_pemrosesan")
def view_pemrosesan():
    if (session.get("preproses") == None):
        view = "/dataset"
    else:
        view = "/pemrosesan"
    return redirect(view)


@app.route("/pemrosesan")
def preprocessing():
    connection = connectDB()
    pilihData = session.get("pilih")
    if (pilihData == "dataset"):
        kondisi = 0
        view = redirect("/dataset")
    else:
        kondisi = 2

    namaFile = "static/files/#tweet_label_dikit.xlsx"
    DataTweet = pd.read_excel(namaFile)
    #selectData, cursor = selectDB(connection, pilihData)
    # cursor.close()
    #DataTweet = pd.DataFrame(selectData, columns=["No", "Tweet", "Label"])
    DataTweet2 = pd.DataFrame()
    # ============== Start Processing Text
    print("\n##-------- Mulai Proses Preprocessing --------##\n")
    print('\n...... Proses Casefolding lowercase, hapus URL...... ')
    DataTweet['casefolding'] = DataTweet['Tweet'].apply(
        preProcessingModul.lower)
    DataTweet['removeURL'] = DataTweet['casefolding'].apply(
        preProcessingModul.removeURLemoji)
    print(DataTweet[['removeURL']].head(2))
    # ==== Tokenisasi : memisahkan kata dalam kalimat
    print('\n...... Tokenisasi ...... ')
    DataTweet['Tokenisasi'] = DataTweet['removeURL'].apply(
        preProcessingModul.tokenize)
    print(DataTweet[['Tokenisasi']].head(2))
    print('\n...... Proses Cleaning hapus angka dan simbol...... ')
    DataTweet['Cleanings'] = DataTweet['Tokenisasi'].apply(
        preProcessingModul.hapus_simbolAngka)
    DataTweet2['Cleanings'] = DataTweet['Tokenisasi'].apply(
        preProcessingModul.hapus_simbolAngka)
    # ============== Normalisasi: kata gaul, singkatan jadi kata baku
    print('\n...... Proses Normalisasi ...... ')
    DataTweet['Normalisasi'] = DataTweet['Cleanings'].apply(
        preProcessingModul.normalisasi)
    print(DataTweet[['Normalisasi']].head(2))

    # ==== Stopword Removal : hapus kata yang tidak terlalu penting
    print('\n...... Proses Stopword Removal ...... ')
    tambahan, cursor3 = selectDB(connection, 'stopword')
    cursor3.close()
    DataTweet['Stopword'] = DataTweet['Normalisasi'].apply(
        preProcessingModul.delstopword, tambah=tambahan)
    print(DataTweet[['Stopword']].head(6))
    # ==== Stemming : mengurangi dimensi fitur kata/term
    print('\n................ Proses Stemming ................ ')
    DataTweet['Stemmed'] = DataTweet['Stopword'].apply(
        preProcessingModul.stemming)
    print(DataTweet['Stemmed'].head(3))

    DataTweet = DataTweet.drop(columns=['Cleanings'])
    DataTweet['Cleanings'] = DataTweet2['Cleanings']
    print('\n==========')
    DataTweet['newTweet'] = DataTweet['Stemmed'].apply(
        preProcessingModul.listokalimat)
    data = {"preproses": session.get("preproses"), "tfidf": session.get("tfidf"),
            "train_model": session.get("train_model")}
    view = render_template(
        'pemrosesan.html', dataset=DataTweet, data=data, kondisi=kondisi)
    return view


@app.route("/view_tf_idf")
def view_tf_idf():
    if (session.get("tfidf") == False):
        view = redirect("/view_pemrosesan")
    else:
        view = redirect("/tfidf")
    return view


def tf_idf(pilihData):
    connection = connectDB()
    if (pilihData == "dataset"):
        Stemming, cursor = selectDB(connection, 'preprocessing')
        DataStemming = pd.DataFrame(
            Stemming, columns=["ID", "removeURL", "Tokenisasi", "Cleaning", "Normalisasi", "Stopword", "Stemming"])
        print(DataStemming)

    else:
        redirect("/dataset")
    # ====================== lakukan TF-IDF
    print('\n................ Hitung TF-IDF ................ ')
    tfidf_vect = TfidfVectorizer(use_idf=True)
    vect_docs = tfidf_vect.fit_transform(DataStemming['Stemming'])
    features_names = tfidf_vect.get_feature_names_out()
    dense = vect_docs.todense()
    alist = dense.tolist()
    print('\n================')
    newData = pd.DataFrame(alist, columns=features_names)
    session["tfidf"] = True
    """
    if (session["uname"] == "admin" and pilihData == "dataset"):
        pickle.dump(tfidf_vect.vocabulary_, open(
            "static/files/pickle_feature.pkl", "wb"))
    """
    return vect_docs, features_names, newData


@app.route("/tfidf")
def gotfidf():
    pilihData = session.get("pilih")
    preproses = session.get("preproses")
    train_model = session.get("train_model")
    if (preproses == False):
        view = redirect("/view_pemrosesan")
    else:
        if (train_model == True):
            flash(
                "Klasifikasi telah dilakukan. Jika dilakukan kembali membutuhkan waktu yang lama.")
        else:
            flash(
                "Proses Klasifikasi membutuhkan waktu yang lama. Sehingga hanya ditampilkan")
        jumlahDocs = 6
        vect_docs, features_names, newData = tf_idf(pilihData)
        print(newData)
        jmlfitur = len(features_names)
        print("Jumlah Fitur:", jmlfitur, "\nSorting Ranking")
        ranke = pd.DataFrame(vect_docs.todense(),
                             columns=features_names).iloc[:jumlahDocs]
        term_document_matrix = ranke.T
        term_document_matrix.columns = [
            'Dokumen'+str(i+1) for i in range(jumlahDocs)]
        # print(term_document_matrix)

        termTampil = 25
        term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)
        term_document_matrix = term_document_matrix.sort_values(
            by='total_count', ascending=False)[:termTampil]
        rankingTFIDF = term_document_matrix.drop(columns=['total_count'])
        rankingTFIDF = rankingTFIDF.apply(preProcessingModul.bulatkan)
        print(rankingTFIDF)
        data = {"preproses": session.get("preproses"), "tfidf": session.get("tfidf"),
                "train_model": session.get("train_model")}
        view = render_template('tfidf.html', tables=rankingTFIDF.to_html(classes="data-tables datatable-primary", table_id="tfidf"),
                               data=data, jmlfitur=jmlfitur)
    return view


@app.route("/view_klasifikasi")
def view_klasifikasi():
    #global DataTweet, newData
    if (session.get("train_model") == False):
        view = redirect("/view_pemrosesan")
    else:
        view = redirect("/tfidf")
    return view


@app.route("/klasifikasi")
def klasifikasi():
    rata2akurasi = session.get("rata2akurasi")
    kfold = session.get("kfold")
    if session.get("train_model") == False:
        kosong = True
        print("\nBelum dilakukan Klasifikasi")
        view = redirect("/view_tf_idf")
    else:
        kosong = False
        data = {"preproses": session.get("preproses"), "tfidf": session.get("tfidf"),
                "train_model": session.get("train_model")}
        view = render_template('klasifikasi.html', rata2akurasi=rata2akurasi,
                               data=data, kosong=kosong, kfold=kfold)
    return view


@app.route("/goklasifikasi")
def goklasifikasi():
    connection = connectDB()
    pilihData = session.get("pilih")
    # datatweet
    selectData, cursor = selectDB(connection, pilihData)
    DataTweet = pd.DataFrame(selectData, columns=["No", "Tweet", "Label"])
    # tfidf
    vect_docs, features_names, newData = tf_idf(pilihData)
    kfold = []
    x = newData.iloc[:]
    y = DataTweet["Label"]

    if (session.get("train_model") == None):
        print('\n================ Model ================ ')
        # ============================ K-fold Start
        print('\nK - Fold Cross Validation')
        k = 5
        kf = KFold(n_splits=k)

        temp_akurasi = []
        temp_pres = []
        temp_recall = []
        temp_f1 = []
        temp_model = []
        it = 1
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for train_index, test_index in kf.split(x):
            X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            print('\n================ Fold ke -', it)
            #print('Data Train\n',X_train)
            #print('Data Test\n',X_test)
            sm = SMOTE(random_state=29)
            x_oversample, y_oversample = sm.fit_resample(X_train, y_train)
            # setelah resampling dengan SMOTE
            jumlah_awal = y_train.shape[0]
            cekLabel_awal = Counter(y_train)
            jumlah_sm = y_oversample.shape[0]
            cekLabel_sm = Counter(y_oversample)
            jumlah_tes = y_test.shape[0]
            print('Jumlah Data latih sebelum SMOTE =', jumlah_awal)
            print('Sarkasme =', cekLabel_awal['Sarkasme'],
                  'BukanSarkas =', cekLabel_awal['BukanSarkas'])
            print('Jumlah Data latih setelah SMOTE =', jumlah_sm)
            print('Sarkasme =', cekLabel_sm['Sarkasme'],
                  'BukanSarkas =', cekLabel_sm['BukanSarkas'])
            print('Jumlah Data Uji =', jumlah_tes)
            baseLearn_svm = SVC(probability=True, kernel='linear')
            model_adaboost = AdaBoostClassifier(
                n_estimators=30, base_estimator=baseLearn_svm, learning_rate=0.5)
            model_adaboost.fit(x_oversample, y_oversample)
            Prediksi = model_adaboost.predict(X_test)
            ceklah = pd.DataFrame(
                columns=['Tweet_Split', 'Label_Split', 'Label_Prediksi'])
            ceklah['newTweet'] = DataTweet['Tweet'].iloc[test_index]
            ceklah['Label'] = DataTweet['Label'].iloc[test_index]
            ceklah['LabelPrediksi'] = Prediksi

            jumlahtes = ceklah.shape[0]
            positif = "Sarkasme"
            negatif = "BukanSarkas"
            for i in range(jumlahtes):
                cek = ceklah.iloc[i]
                if (cek.Label == positif and cek.LabelPrediksi == positif):
                    TP += 1
                elif (cek.Label == positif and cek.LabelPrediksi == negatif):
                    FP += 1
                elif (cek.Label == negatif and cek.LabelPrediksi == negatif):
                    TN += 1
                elif (cek.Label == negatif and cek.LabelPrediksi == positif):
                    FN += 1
            print("(TP) TruePositif :", TP, "\n(FP) FalsePositif :", FP,
                  "\n(TN) TrueNegatif :", TN, "\n(FN) FalsePNegatif :", FN)
            # akurasi
            akurasi = (TP+TN)/(TP+FP+TN+FN)
            # presisi
            cek0 = TP+FP
            if (cek0 == 0):
                presisi = 0
            else:
                presisi = TP/(TP+FP)
            # recall
            cek1 = TP+FN
            if (cek1 == 0):
                recal = 0
            else:
                recal = TP/(TP+FN)

            if (presisi == 0 and recal == 0):
                f1 = 0
            else:
                f1 = (2*presisi*recal)/(presisi+recal)
            hasil_akurasi = round(akurasi*100, 3)
            hasil_pres = round(presisi*100, 3)
            hasil_recal = round(recal*100, 3)
            hasil_f1 = round(f1*100, 3)

            temp_akurasi.append(hasil_akurasi)
            temp_pres.append(hasil_pres)
            temp_recall.append(hasil_recal)
            temp_f1.append(hasil_f1)
            print('Akurasi Fold ke -', it, '=', hasil_akurasi)
            print('Presisi Fold ke -', it, '=', hasil_pres)
            print('Recall Fold ke -', it, '=', hasil_recal)
            print('F1-Measure Fold ke -', it, '=', hasil_f1)
            temp_model.append(model_adaboost)
            kfold.append([jumlahtes, TP, FP, TN, FN, hasil_akurasi,
                          hasil_pres, hasil_recal, hasil_f1])
            it = it+1
            TP = 0
            FP = 0
            TN = 0
            FN = 0
        rata2 = 0
        rata2rec = 0
        rata2pre = 0
        rata2f1 = 0
        for x in range(len(temp_akurasi)):
            rata2 = rata2+temp_akurasi[x]
        for x in range(len(temp_recall)):
            rata2rec = rata2rec+temp_recall[x]
        for x in range(len(temp_pres)):
            rata2pre = rata2pre+temp_pres[x]
        for x in range(len(temp_f1)):
            rata2f1 = rata2f1+temp_f1[x]

        hasilrata = round(rata2/k, 3)
        print("rata-rata akurasi", hasilrata)
        print("rata-rata presisi", round(rata2rec/k, 3))
        print("rata-rata recall", round(rata2pre/k, 3))
        print("rata-rata f1", round(rata2f1/k, 3))

        # ============================ K-fold End
        session['kfold'] = kfold
        session["rata2akurasi"] = hasilrata
        session['train_model'] = True

    return redirect('/klasifikasi')

@app.route("/history")
def klasifikasi2():
    data = {"preproses": session.get("preproses"), "tfidf": session.get("tfidf"),
            "train_model": session.get("train_model")}
    return render_template('history.html', data=data)


if __name__ == "__main__":
    global vect_docs, features_names, stopwordlist, jmlStopword, savedmodelTF, savedmodelTrain
    connection = connectDB()
    DataTweet = pd.DataFrame()
    DataTweet2 = pd.DataFrame()
    DataCrawl = pd.DataFrame()
    newData = pd.DataFrame()
    #rankingTFIDF = pd.DataFrame()
    #rata2akurasi = 0
    #kfold = []
    #app.run(host='127.0.0.1', port=5002, debug=True)
    app.run(host='0.0.0.0', port=5002, debug=True)
