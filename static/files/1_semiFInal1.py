# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:26:49 2021

@author: AhmadYazidMunif
"""
import sys
import os
import pandas as pd
import preProcessingModul
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from collections import Counter
from sklearn import model_selection
from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from pandas import ExcelWriter
import matplotlib.pyplot as plt
import pickle
import os.path
"""
modules = dir()

print(modules)
#============== Read Data Input
#namaFile="test2.csv"
namaFile="#tweet_label.csv"

DataTweet2 = pd.read_csv(namaFile)
DataTweet = DataTweet2.sample(frac=1).reset_index()
print(DataTweet)
jumlahData=DataTweet["Tweet"].shape[0] #Jumlah Data
jmlSarkas=DataTweet[DataTweet.Label=="Sarkasme"].shape[0]
jmlNonSarkas=DataTweet[DataTweet.Label=="BukanSarkas"].shape[0]
d={'Jumlah': [jumlahData], 'Sarkasme': [jmlSarkas],'BukanSarkas':[jmlNonSarkas]}

cekJumlah = pd.DataFrame(data=d, columns=['Sarkasme','BukanSarkas'])

print('\n...... Jumlah Data ...... ')
print("Total:",jumlahData)
print(cekJumlah)
#ax = cekJumlah.plot.bar(rot=0)
#============== Start Processing Text
print("\n##-------- Mulai Proses Preprocessing --------##\n")
print('\n...... Proses Casefolding lowercase, hapus URL...... ')
#df_preprocess["RemoveURL"] = DataTweet['Tweet'].apply(preProcessingModul.lowRemoveURL)
DataTweet['casefolding'] = DataTweet['Tweet'].apply(preProcessingModul.lower)
DataTweet['removeURL'] = DataTweet['casefolding'].apply(preProcessingModul.removeURLemoji)
print(DataTweet)

#==== Tokenisasi : memisahkan kata dalam kalimat
print('\n...... Tokenisasi ...... ')
DataTweet['Tokenisasi'] = DataTweet['removeURL'].apply(preProcessingModul.tokenize)
#df_preprocess['Tokenisasi'] = df_preprocess["RemoveURL"].apply(preProcessingModul.tokenize)
print(DataTweet[['Tokenisasi']].head(2))

print('\n...... Proses Casefolding2 hapus angka dan simbol...... ')
#DataTweet['hapus2'] = DataTweet['Tokenisasi'].apply(preProcessingModul.angkadua)
#DataTweet['hapus2'] = DataTweet['hapus2'].apply(preProcessingModul.hapus_hurufganda)
DataTweet['Cleaning'] = DataTweet['Tokenisasi'].apply(preProcessingModul.hapus_simbolAngka)
#df_preprocess['hapus2'] = df_preprocess['Tokenisasi'].apply(preProcessingModul.angkadua)
#df_preprocess['hapus2'] = df_preprocess['hapus2'].apply(preProcessingModul.hapus_hurufganda)
#df_preprocess['casefolding2'] = df_preprocess['hapus2'].apply(preProcessingModul.hapus_simbolAngka)
DataTweet2=pd.DataFrame()
DataTweet2['Cleaning']=DataTweet['Cleaning']
print(DataTweet2[['Cleaning']].head(2))
#============== Normalisasi: kata gaul, singkatan jadi kata baku
print('\n...... Proses Normalisasi ...... ')
#normalisasi_dict = normal_term() #import excel
DataTweet['Normalisasi'] = DataTweet['Cleaning'].apply(preProcessingModul.normalisasi)
#df_preprocess['Normalisasi'] = df_preprocess['casefolding2'].apply(preProcessingModul.normalisasi)
print(DataTweet[['Normalisasi']].head(2))

#==== Stopword Removal : hapus kata yang tidak terlalu penting
print('\n...... Proses Stopword Removal ...... ')
#list_stopwords = daftarStopword()
DataTweet['Stopword'] = DataTweet['Normalisasi'].apply(preProcessingModul.delstopwordID)
#df_preprocess['Stopword'] = df_preprocess['Normalisasi'].apply(preProcessingModul.delstopwordID)
print(DataTweet[['Stopword']].head(6))


#==== Stemming : mengurangi dimensi fitur kata/term
print('\n................ Proses Stemming ................ ')

DataTweet['Stemmed'] = DataTweet['Stopword'].apply(preProcessingModul.stemming)
print(DataTweet['Stemmed'].head(3))
DataTweet['newTweet'] = DataTweet['Stemmed'].apply(preProcessingModul.listokalimat)
'''
DataTweet['newTweet'] = DataTweet['Stopword'].apply(preProcessingModul.listokalimat)
'''
DataTweet.to_excel('_preprocessing.xlsx', index=False)
excDataTweet = pd.read_excel('_preprocessing.xlsx', index_col=None)
print('\n==========')
#print(DataTweet['newTweet'].head(2))
print(excDataTweet)


#====================== lakukan TF-IDF
print('\n................ Hitung TF-IDF ................ ')
tfidf_vect = TfidfVectorizer()
vect_docs = tfidf_vect.fit_transform(excDataTweet['newTweet'])
print(vect_docs)
features_names = tfidf_vect.get_feature_names_out()
print(len(features_names))
datane = []
means=vect_docs.mean(axis=0)
for col, term in enumerate(features_names):
    datane.append( (term, means[0,col] ))

ranking = pd.DataFrame(datane, columns=['term','rata2bobot'])
ranking = ranking.sort_values('rata2bobot', ascending=False)
print(ranking.head(7))

dense = vect_docs.todense()
alist = dense.tolist()
print('\n================')
newData = pd.DataFrame(alist,columns=features_names)
#df = pd.DataFrame(alist, columns=features_names)
print(newData)




x = newData.iloc[:]
#print(x)
y= excDataTweet["Label"]
#print(y)

print('\n================ Model ================ ')
print('\n================ Pembagian data Training dan Testing ================ ')
#============================ K-fold Start
print('\nK - Fold Cross Validation')
k=5
kf = KFold(n_splits=k)
print("fold-berjumlah:",k)
kfold=[]
temp_akurasi = []
temp_pres = []
temp_recall = []
temp_f1 = []
temp_model=[]

it=1
TP = 0
FP = 0
TN = 0
FN = 0

for train_index , test_index in kf.split(x):
    X_train , X_test = x.iloc[train_index,:],x.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    print(len(X_train.columns))
    print('\n================ Fold ke -',it)
    #print('Data Train\n',X_train)
    #print('Data Test\n',X_test)
    sm = SMOTE(sampling_strategy="minority",random_state=1,k_neighbors=5)
    x_oversample, y_oversample = sm.fit_resample(X_train, y_train)
    #setelah resampling dengan SMOTE
    jumlah_awal = y_train.shape[0]
    cekLabel_awal = Counter(y_train)
    jumlah_sm = y_oversample.shape[0]
    cekLabel_sm = Counter(y_oversample)
    jumlah_tes = y_test.shape[0]
    print('Jumlah Data latih sebelum SMOTE =', jumlah_awal)
    print('Sarkasme =', cekLabel_awal['Sarkasme'],'BukanSarkas =',cekLabel_awal['BukanSarkas'])
    print('Jumlah Data latih setelah SMOTE =', jumlah_sm)
    print('Sarkasme =', cekLabel_sm['Sarkasme'],'BukanSarkas =',cekLabel_sm['BukanSarkas'])
    print('Jumlah Data Uji =', jumlah_tes)
    #================== Metode klasifikasi
    
    baseLearn_svm=SVC(probability=True,kernel='linear',max_iter=30)
    baseLearn_svm.fit(x_oversample,y_oversample)
    Prediksi = baseLearn_svm.predict(X_test)
    temp_model.append(baseLearn_svm)
    '''
    model_adaboost =AdaBoostClassifier(n_estimators=30, base_estimator=baseLearn_svm,learning_rate=0.5)
    model_adaboost.fit(x_oversample,y_oversample)
    Prediksi = model_adaboost.predict(X_test)
    temp_model.append(model_adaboost)
    '''
    #=================
    ceklah = pd.DataFrame(columns=['Tweet_Split','Label_Split','Label_Prediksi'])
    ceklah['newTweet'] = DataTweet['Tweet'].iloc[test_index]
    ceklah['Label'] = DataTweet['Label'].iloc[test_index]
    ceklah['LabelPrediksi'] = Prediksi
    
    jumlahtes = ceklah.shape[0]
    positif="Sarkasme"
    negatif="BukanSarkas"
    for i in range(jumlahtes):
        cek=ceklah.iloc[i]
        if (cek.Label==positif and cek.LabelPrediksi==positif):
            TP+=1
        elif(cek.Label==positif and cek.LabelPrediksi==negatif):
            FN+=1
        elif(cek.Label==negatif and cek.LabelPrediksi==negatif):
            TN+=1
        elif(cek.Label==negatif and cek.LabelPrediksi==positif):
            FP+=1
    
    print("(TP) TruePositif :",TP,"\n(FP) FalsePositif :",FP,"\n(TN) TrueNegatif :",TN,"\n(FN) FalseNegatif :",FN)
    
    akurasi=(TP+TN)/(TP+FP+TN+FN)
    presisi=TP/(TP+FP)
    recal=TP/(TP+FN)
    if(presisi==0 and recal==0):
        f1=0
    else:
        f1=(2*presisi*recal)/(presisi+recal)
    hasil_akurasi = round(akurasi*100,3)
    hasil_pres=round(presisi*100,3)
    hasil_recal=round(recal*100,3)
    hasil_f1=round(f1*100,3)
    
    
    temp_akurasi.append(hasil_akurasi)
    temp_pres.append(hasil_pres)
    temp_recall.append(hasil_recal)
    temp_f1.append(hasil_f1)
    print('Akurasi Fold ke -',it,'=',hasil_akurasi)
    print('Presisi Fold ke -',it,'=',hasil_pres)
    print('Recall Fold ke -',it,'=',hasil_recal)
    print('F1-Measure Fold ke -',it,'=',hasil_f1)
    kfold.append([jumlahtes,TP,FP,TN,FN,hasil_akurasi,hasil_pres,hasil_recal,hasil_f1])
    it=it+1
    TP = 0
    FP = 0
    TN = 0
    FN = 0

rata2=0
rata2rec=0
rata2pre=0
rata2f1=0
for x in range(len(temp_akurasi)):
    rata2=rata2+temp_akurasi[x]
print("rata-rata akurasi=",round(rata2/k,3))
print(temp_model)
maxacc=max(temp_akurasi)
print("MAX akurasi",maxacc)

maxi = temp_akurasi.index(maxacc)
modele=temp_model[maxi]
"""
"""
# Save to file in the current working directory
fileakurasi="_akurasi.txt"
pkl_filename = "pickle_model.pkl"

akurasiModel = open(fileakurasi, "w+")
a=akurasiModel.read()
print(a)

if((a=="") or (float(a)<maxacc)):
    akurasiModel.write(str(maxacc))
    akurasiModel.close()
    '''
    if (os.path.exists(pkl_filename)):
        os.remove(pkl_filename)
    '''
    with open(pkl_filename,'wb') as file:
        pickle.dump(modele,file)
    
    
else:
    akurasiModel.close()

pickle.dump(tfidf_vect.vocabulary_, open("pickle_feature.pkl", "wb"))
"""

it=1
TP = 0
FP = 0
TN = 0
FN = 0
# Load from file
with open("pickle_model.pkl", 'rb') as file:
    pickle_model = pickle.load(file)

namaFile3="#tweet_label1.csv"

DataTweet3= pd.read_csv(namaFile3)
jumlahData3=DataTweet3["Tweet"].shape[0] #Jumlah Data
print('\n...... Jumlah Data ...... ')
print("Total:",jumlahData3)
#============== Start Processing Text
print("\n##-------- Mulai Proses Preprocessing --------##\n")
print('\n...... Proses Casefolding lowercase, hapus URL...... ')
DataTweet3['casefolding'] = DataTweet3['Tweet'].apply(preProcessingModul.lower)
DataTweet3['removeURL'] = DataTweet3['casefolding'].apply(preProcessingModul.removeURLemoji)
print(DataTweet3)

#==== Tokenisasi : memisahkan kata dalam kalimat
print('\n...... Tokenisasi ...... ')
DataTweet3['Tokenisasi'] = DataTweet3['removeURL'].apply(preProcessingModul.tokenize)
print(DataTweet3[['Tokenisasi']].head(2))

print('\n...... Proses Casefolding2 hapus angka dan simbol...... ')
DataTweet3['Cleaning'] = DataTweet3['Tokenisasi'].apply(preProcessingModul.hapus_simbolAngka)
DataTweet4=pd.DataFrame()
DataTweet4['Cleaning']=DataTweet3['Cleaning']
print(DataTweet4[['Cleaning']].head(2))
#============== Normalisasi: kata gaul, singkatan jadi kata baku
print('\n...... Proses Normalisasi ...... ')
DataTweet3['Normalisasi'] = DataTweet3['Cleaning'].apply(preProcessingModul.normalisasi)
print(DataTweet3[['Normalisasi']].head(2))

#==== Stopword Removal : hapus kata yang tidak terlalu penting
print('\n...... Proses Stopword Removal ...... ')
DataTweet3['Stopword'] = DataTweet3['Normalisasi'].apply(preProcessingModul.delstopwordID)
print(DataTweet3[['Stopword']].head(6))

#==== Stemming : mengurangi dimensi fitur kata/term
print('\n................ Proses Stemming ................ ')
DataTweet3['newTweet'] = DataTweet3['Stopword'].apply(preProcessingModul.listokalimat)

#====================== lakukan TF-IDF
print('\n................ Hitung TF-IDF ................ ')
savedtfidf = pickle.load(open("pickle_feature.pkl", 'rb'))
vectorizer2 = TfidfVectorizer(vocabulary=savedtfidf)
vect_docs2 = vectorizer2.fit_transform(DataTweet3['newTweet'])
features_names2 = vectorizer2.get_feature_names_out()
print(vect_docs2)

dense2 = vect_docs2.todense()
alist2 = dense2.tolist()
print('\n================')
newData2 = pd.DataFrame(alist2,columns=features_names2)
print(newData2)
hasil=pickle_model.predict(newData2)
DFpredict = pd.DataFrame(hasil,columns=["Prediksi"])
gabungkan = pd.concat([DataTweet3[["Tweet","Label"]],DFpredict], axis=1)
print(gabungkan[["Label","Prediksi"]].head(10))

implementasi = gabungkan.shape[0]
positif="Sarkasme"
negatif="BukanSarkas"
for i in range(implementasi):
    cek=gabungkan.iloc[i]
    if (cek.Label==positif and cek.Prediksi==positif):
        TP+=1
    elif(cek.Label==positif and cek.Prediksi==negatif):
        FN+=1
    elif(cek.Label==negatif and cek.Prediksi==negatif):
        TN+=1
    elif(cek.Label==negatif and cek.Prediksi==positif):
        FP+=1

print("(TP) TruePositif :",TP,"\n(FP) FalsePositif :",FP,"\n(TN) TrueNegatif :",TN,"\n(FN) FalseNegatif :",FN)

akurasi=(TP+TN)/(TP+FP+TN+FN)
presisi=TP/(TP+FP)
recal=TP/(TP+FN)
if(presisi==0 and recal==0):
    f1=0
else:
    f1=(2*presisi*recal)/(presisi+recal)
hasil_akurasi = round(akurasi*100,3)
hasil_pres=round(presisi*100,3)
hasil_recal=round(recal*100,3)
hasil_f1=round(f1*100,3)

print(hasil_akurasi)
#print("rata-rata akurasi",round(rata2/k,3))
#print(kfold)
#============================ K-fold End
