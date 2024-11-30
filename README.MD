# LAPORAN PROYEK MACHINE LEARNING - Ilham Julian Efendi
## Domain Proyek: Kesehatan
Stroke adalah salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Menurut data Organisasi Kesehatan Dunia (WHO), setiap tahun sekitar 15 juta orang di seluruh dunia mengalami stroke, dengan 5 juta di antaranya meninggal dan 5 juta lainnya mengalami kecacatan permanen (WHO, 2021). Penyakit ini dapat dicegah jika risiko terkena stroke dapat dideteksi lebih awal. Salah satu cara untuk meningkatkan deteksi dini adalah dengan menggunakan teknologi berbasis data, seperti machine learning, yang memungkinkan analisis prediktif secara cepat dan akurat.

Machine learning menawarkan solusi inovatif untuk mendeteksi risiko stroke dengan menganalisis data karakteristik pasien, termasuk usia, riwayat penyakit, gaya hidup, dan status kesehatan lainnya. Berbagai algoritma machine learning seperti Random Forest dan Support Vector Machine (SVM) dapat digunakan untuk mengidentifikasi individu yang memiliki risiko tinggi terkena stroke. Prediksi ini memungkinkan para profesional kesehatan untuk mengambil langkah pencegahan yang tepat dan menyelamatkan lebih banyak nyawa.

Mengapa Masalah Ini Perlu Diselesaikan?
Beban Ekonomi dan Sosial: Stroke tidak hanya berdampak pada kesehatan individu, tetapi juga membawa beban ekonomi yang signifikan. Menurut studi oleh American Stroke Association, biaya perawatan stroke di Amerika Serikat mencapai $46 miliar per tahun, termasuk biaya pengobatan dan hilangnya produktivitas akibat kecacatan (ASA, 2022).
Keterbatasan Diagnostik Tradisional: Pendekatan tradisional dalam mendiagnosis stroke sering kali membutuhkan waktu lama dan mengandalkan sumber daya medis yang terbatas. Dengan pendekatan berbasis data, risiko stroke dapat diidentifikasi lebih awal, bahkan sebelum gejala muncul.
Pencegahan Lebih Efektif: Deteksi dini memungkinkan pemberian terapi atau intervensi gaya hidup yang lebih efektif untuk mengurangi risiko terkena stroke, terutama pada individu dengan riwayat hipertensi atau diabetes.
Bagaimana Masalah Ini Diselesaikan?
Proyek ini menggunakan dataset yang relevan dari Kaggle (Stroke Prediction Dataset) untuk membangun model prediksi risiko stroke. Dengan memanfaatkan algoritma machine learning, seperti Random Forest dan SVM, proyek ini akan menghasilkan model prediksi yang akurat. Selain itu, teknik optimasi seperti Hyperparameter Tuning akan digunakan untuk meningkatkan performa model. Prediksi yang dihasilkan akan membantu profesional kesehatan untuk:

Mengidentifikasi pasien yang berisiko tinggi.
Merencanakan intervensi dini berdasarkan hasil prediksi.
Mengurangi angka kematian dan kecacatan akibat stroke.
Referensi
Organisasi Kesehatan Dunia (WHO): "The Top 10 Causes of Death" (WHO, 2021).
American Stroke Association (ASA): "Stroke Statistics" (ASA, 2022).
Chollet, Francois: "Deep Learning with Python, Second Edition" - Membahas bagaimana machine learning digunakan dalam bidang kesehatan untuk prediksi penyakit.
Goyal, M., et al.: "Stroke prevention strategies: A global perspective" (The Lancet, 2020).
Dengan menyelesaikan masalah ini, proyek ini tidak hanya membantu menyelamatkan nyawa tetapi juga memberikan kontribusi penting dalam mengurangi beban ekonomi dan sosial yang disebabkan oleh stroke.

## Business Understanding
### Problem Statements
Stroke adalah salah satu penyebab utama kematian dan kecacatan di dunia. Identifikasi dini terhadap risiko stroke dapat membantu dalam memberikan intervensi yang cepat dan tepat. Namun, terdapat beberapa tantangan yang harus diatasi:

Pernyataan Masalah 1:
Bagaimana memanfaatkan data pasien, seperti riwayat hipertensi, penyakit jantung, dan gaya hidup, untuk memprediksi risiko stroke secara akurat?

Pernyataan Masalah 2:
Bagaimana memilih dan mengoptimalkan algoritma machine learning yang tepat untuk menghasilkan model prediksi dengan performa tinggi?

Pernyataan Masalah 3:
Bagaimana menangani distribusi data yang tidak seimbang, di mana jumlah pasien dengan stroke jauh lebih sedikit dibandingkan yang tidak mengalami stroke?

### Goals
Proyek ini bertujuan untuk menjawab pertanyaan-pertanyaan di atas dengan tujuan utama sebagai berikut:

Jawaban Pernyataan Masalah 1:
Membuat model prediksi berbasis machine learning menggunakan data karakteristik pasien, seperti usia, riwayat penyakit, dan status merokok, untuk mendeteksi risiko stroke.

Jawaban Pernyataan Masalah 2:
Menerapkan dan mengevaluasi berbagai algoritma machine learning, seperti Random Forest dan Support Vector Machine (SVM), untuk memilih model yang memberikan hasil prediksi terbaik.

Jawaban Pernyataan Masalah 3:
Mengatasi distribusi data yang tidak seimbang menggunakan teknik seperti oversampling atau SMOTE (Synthetic Minority Oversampling Technique) untuk meningkatkan kemampuan model dalam memprediksi kelas minoritas (stroke).

### Solution Statements
Untuk mencapai tujuan tersebut, solusi yang diusulkan mencakup langkah-langkah berikut:

Menggunakan Dua Algoritma Machine Learning:

Random Forest: Algoritma ensemble yang memanfaatkan decision tree untuk menghasilkan model yang stabil dan akurat.
Support Vector Machine (SVM): Algoritma yang efektif dalam memisahkan data kelas dengan garis batas (hyperplane) yang optimal.
Improvement dengan Hyperparameter Tuning:

Menggunakan GridSearchCV untuk menemukan kombinasi parameter terbaik yang meningkatkan performa model, seperti:
Random Forest: n_estimators, max_depth, dan min_samples_split.
SVM: C, gamma, dan jenis kernel (rbf, poly, atau linear).
Penanganan Ketidakseimbangan Data:

Menggunakan SMOTE untuk menyeimbangkan distribusi kelas, sehingga model dapat mempelajari pola yang lebih baik untuk prediksi risiko stroke.
Evaluasi dengan Metrik yang Tepat:

Menggunakan metrik seperti accuracy, precision, recall, dan F1-score untuk mengevaluasi performa model, dengan fokus pada meningkatkan kemampuan model dalam memprediksi kelas minoritas (stroke).
Output yang Diharapkan
Dengan solusi di atas, proyek ini diharapkan dapat menghasilkan:

Model prediktif yang akurat untuk mendeteksi risiko stroke berdasarkan data karakteristik pasien.
Wawasan tambahan bagi profesional kesehatan untuk mengambil langkah pencegahan yang lebih cepat dan efektif.
Alat pendukung berbasis data untuk membantu pengambilan keputusan dalam menangani pasien dengan risiko tinggi.

## Data Understanding
Dalam tahapan Data Understanding, fokus utama adalah memahami dataset yang digunakan untuk proyek prediksi stroke. Dataset diambil dari Kaggle, yang terdiri dari 5110 entri dengan 11 variabel. Dataset ini memuat berbagai informasi tentang karakteristik pasien seperti jenis kelamin, usia, status kesehatan, dan gaya hidup. Variabel target (stroke) memiliki dua kategori:

0: Tidak mengalami stroke
1: Mengalami stroke

Berikut deskripsi dari setiap fitur dalam dataset:

| No | Variabel	| Deskripsi |
| -- | -------- | --------- |
| 1	| gender	| Jenis kelamin pasien (Male/Female/Other) |
| 2 |	age | Usia pasien |
| 3	| hypertension |	Riwayat hipertensi (1: Ya, 0: Tidak) |
| 4	| heart_disease	| Riwayat penyakit jantung (1: Ya, 0: Tidak) |
| 5 |	ever_married	| Status pernikahan pasien (Yes/No) |
| 6 |	work_type	| Jenis pekerjaan (Private/Self-employed/Govt_job/Children/Never_worked) |
| 7	| Residence_type	| Tipe tempat tinggal (Urban/Rural) |
| 8	| avg_glucose_level |	Rata-rata tingkat glukosa dalam darah |
| 9	| bmi	| Indeks Massa Tubuh |
| 10	| smoking_status	| Status merokok (never smoked/formerly smoked/smokes/unknown) |
| 11	| stroke	| Target variabel (1: Mengalami stroke, 0: Tidak mengalami stroke) |

### Informasi Data
Jumlah Data: 5110 entri
Fitur: 11 kolom
Variabel Target: stroke
Missing Values: Terdapat nilai hilang pada kolom bmi.

Memvisualisasikan data menggunakan boxplot untuk fitur numerik: [age,hypertension, heart_disease, avg_glucose_level, bmi]

![Visualisasi Boxplot Age](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding1.png?raw=true)
![Visualisasi Boxplot hypertension](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding2.png?raw=true)
![Visualisasi Boxplot heart_disease](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding3.png?raw=true)
![Visualisasi Boxplot avg_glucose_level](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding4.png?raw=true)
![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding5.png?raw=true)

Menganalisa data menggunakan Univariate Analysis Membagi fitur numerik dan kategorik yang terdapat pada dataset
Melakukan analisa fitur kategori

Count  Percent
gender                
Female   2897     59.0
Male     2011     41.0
Other       1      0.0
<Axes: title={'center': 'gender'}, xlabel='gender'>

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding6.png?raw=true)

Menganalisa fitur gender dimana gender other sudah dihilangkan

    Count  Percent
gender                
Female   2897     59.0
Male     2011     41.0
<Axes: title={'center': 'gender'}, xlabel='gender'>

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding7.png?raw=true)

menganalisa fitur ever maried

             Count  Percent
ever_married                
Yes            3204     65.3
No             1704     34.7
<Axes: title={'center': 'ever_married'}, xlabel='ever_married'>

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding8.png?raw=true)

Menganalisa fitur work_type

    Count  Percent
work_type                    
Private         2810     57.3
Self-employed    775     15.8
children         671     13.7
Govt_job         630     12.8
Never_worked      22      0.4
<Axes: title={'center': 'work_type'}, xlabel='work_type'>

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding9.png?raw=true)

Menganalisa feature residence type

Count  Percent
Residence_type                
Urban            2490     50.7
Rural            2418     49.3
<Axes: title={'center': 'Residence_type'}, xlabel='Residence_type'>

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding10.png?raw=true)

Menganalisa feature smoking status

 Count  Percent
smoking_status                 
never smoked      1852     37.7
Unknown           1483     30.2
formerly smoked    836     17.0
smokes             737     15.0
<Axes: title={'center': 'smoking_status'}, xlabel='smoking_status'>

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding11.png?raw=true)

Menganalisa data menggunakan Multivariate Analysis
![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding12.png?raw=true)

Menampilkan Plot Pair fitur numerik

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding13.png?raw=true)

Melakukan pengamatan terhadap tingkat korelasi dengan menggunakan matrik korelasi pada tiap fitur

![Visualisasi Boxplot bmi](https://github.com/iqkyu5566/strokeprediction/blob/main/img/dataunderstanding14.png?raw=true)

## Data Preparation
Setelah melakukan proses Data Understanding dan analisis awal terhadap dataset, langkah berikutnya adalah mempersiapkan data untuk proses pelatihan model machine learning. Tahapan Data Preparation mencakup penghapusan fitur yang tidak relevan, pembagian dataset, serta transformasi data agar model dapat bekerja secara optimal.

### 1. Penghapusan Variabel Tidak Relevan
Berdasarkan hasil analisis korelasi pada tahap sebelumnya, diketahui bahwa terdapat beberapa variabel yang memiliki korelasi rendah atau tidak signifikan terhadap variabel target (stroke). Oleh karena itu, variabel berikut akan dihapus dari dataset:

- id: Merupakan pengenal unik dan tidak relevan dengan prediksi.
- Nilai yang tidak memiliki kontribusi signifikan.

### 2. Penanganan Missing Value
Pada dataset, terdapat missing value pada kolom bmi. Untuk menangani hal ini, baris yang memiliki nilai NaN pada kolom tersebut dihapus

### 3. Encoding Variabel Kategorik
Variabel kategorik seperti gender, ever_married, work_type, Residence_type, dan smoking_status perlu dikonversi ke bentuk numerik menggunakan One-Hot Encoding agar dapat digunakan oleh algoritma machine learning.

## Modeling
Dalam melakukan pemodelan Stroke Prediction Analytics Klasifikasi saya memilih model klasifikasi karena variabel target berupa kalsifikasi rentang nilai 0 - 1 yang menentukan apakah seseorang akan mengalami stroke atau tidak.

Adapun model klasifikasi yang akan saya pilih adalah Random Forest klasifikasi dan Support Vektor Machine dengan melakukan optimasi pada kedua model tersebut menggunakan Hyperparameter GridSearch.

* Hyperparameter Model Random Forest dengan GridSearch

Parameter yang digunakan untuk optimasi model random forest menggunakan GridSearch yaitu:

  * 'n_estimators': [50, 100, 200]
  * 'max_depth': [None, 10, 20, 30]
  * 'min_samples_split': [2, 5, 10]

Dari parameter diatas akan dicari nilai parameter terbaik menggunakan GridSearch untuk model klasifikasi random forest.
> Hasil parameter terbaik dari Hyperparameter GridSearch yaitu:

  > * 'max_depth': 20
  > * 'min_samples_split': 5
  > * 'n_estimators': 200

> Berikut penjelasan dari proses Hyperparamer Tuning dan GridSearch terhadap model:
  > * Hyperparameter tuning dapat digunakan untuk memastikan performa terbaik dari model yang diterapkan, kita akan menggunakan Hyperparameter Tuning dengan GridSearch. Baik Random Forest maupun SVM memiliki hyperparameter yang dapat mempengaruhi performa model secara signifikan. Misalnya, pada Random Forest, jumlah tree (n_estimators) atau kedalaman maksimum tree (max_depth) perlu dioptimalkan, sedangkan pada SVM, parameter seperti C (regularization) dan kernel (linear, polynomial, atau RBF) harus disesuaikan.
  > * GridSearch adalah metode yang memungkinkan kita untuk menguji kombinasi berbagai nilai hyperparameter dan memilih yang terbaik berdasarkan kinerja model. Dalam proyek ini, GridSearch akan menguji berbagai kombinasi parameter dan mengevaluasi model berdasarkan metrik seperti akurasi, precision, recall, dan F1-score. Dengan melakukan tuning yang tepat, model dapat dioptimalkan untuk memberikan hasil klasifikasi yang lebih baik dan akurat dalam memprediksi kisaran harga ponsel. Keunggulan Hyperparameter Tuning dengan GridSearch adalah dapat meningkatkan performa model, dengan menemukan kombinasi hyperparameter terbaik, model akan bekerja lebih optimal dan memberikan hasil klasifikasi yang lebih akurat. Serta keunngulan lainnya dapat mencegah overfitting, dengan pengaturan hyperparameter yang tepat, kita dapat menghindari overfitting dan memastikan bahwa model dapat bekerja dengan baik pada data baru.
Setelah dilakukan optimasi, model yang terbaik akan dievaluasi menggunakan metrik seperti akurasi, precision, recall, dan F1-score, untuk memastikan bahwa prediksi kisaran harga yang dihasilkan dapat diimplementasikan secara efektif dalam pengambilan keputusan penetapan harga perusahaan.
* Model prediksi dengan algoritma Support Vektor Machine:
```
# Inisialisasi model SVM
model_svm = SVC(kernel='rbf', random_state=42)
model_svm.fit(X_train, y_train)
```
> Support Vector Machine (SVM) adalah algoritma yang bekerja dengan menemukan hyperplane terbaik yang memisahkan kelas-kelas data. Dalam proyek ini, SVM akan digunakan untuk memetakan data fitur ponsel ke dalam ruang berdimensi tinggi, kemudian menemukan garis atau kurva (hyperplane) yang memisahkan ponsel berdasarkan kisaran harga. SVM sangat efektif ketika ada perbedaan yang jelas antara kategori harga, dan algoritma ini mampu bekerja dengan baik bahkan ketika data tidak linear, melalui penggunaan kernel trick. Keunggulan SVM yaitu memiliki keakuratan pada data yang tidak seimbang SVM mampu memberikan hasil klasifikasi yang baik, bahkan dalam kasus di mana data tidak seimbang atau memiliki sedikit kesalahan klasifikasi. Serta keunggulan lainnya yaitu dalam penggunaan kernel trick dimana SVM dapat menangani data yang tidak linear dan membuat model lebih fleksibel untuk berbagai macam distribusi data fitur ponsel.
  
> Berikut merupakan penjelasan terhadap setiap parameter yang digunakan:

> * kernel = Algoritma SVM menggunakan serangkaian fungsi matematika yang didefinisikan sebagai kernel. Fungsi kernel adalah mengambil data sebagai input dan mengubahnya ke dalam bentuk yang dibutuhkan. Algoritma SVM yang berbeda menggunakan berbagai jenis fungsi kernel. Fungsi-fungsi ini dapat memiliki tipe yang berbeda. Misalnya linear, nonlinier, polinomial, fungsi basis radial (RBF), dan sigmoid.
> * random_state = mengontrol random number generator yang digunakan. Parameter ini berupa bilangan integer dan nilainya bebas. Parameter ini bertujuan untuk memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan. Jika tidak ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan tes berbeda. Hal ini berpengaruh terhadap akurasi model ML yang menjadi berbeda tiap kali di-run.

* Confusion Matrix model Support Vektor Machine (SVM):
  > Confusion Matrix digunakan untuk melihat hasil prediksi dari model SVM

* Model optimasi algoritma Support Vektor Machine (SVM) dengan Hyperparameter GridSearch:

Parameter yang digunakan untuk optimasi model SVM menggunakan GridSearch yaitu:

  * 'C': [0.1, 1, 10, 100]
  * 'gamma': [1, 0.1, 0.01, 0.001]
  * 'kernel': ['rbf', 'poly', 'sigmoid']

dari parameter diatas akan dicari nilai parameter terbaik menggunakan GridSearch untuk model klasifikasi SVM.
kemudian akan dilihat kembali confusion matrix setelah optimasi.

> Hasil parameter terbaik dari Hyperparameter GridSearch yaitu:

  > * 'C': 0.1
  > * 'gamma': 1
  > * 'kernel': 'poly'

## Evaluasi
Untuk melihat hasil pelatihan dari masing-masing model klasifikasi dengan menggunakan akurasi pada nilai yang dihasilkan pada setiap model, nilai akurasi menggunakan library dari [sklearn](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html). Selainmelihat nilai akurasi pada proyek ini melakukan visualisasi hasil pelatihan dengan confusion matrix. Berikut merupakan hasil akurasi pada setiap model:

* Evaluasi Hasil Model Random Forest:

          precision  recall  f1-score   support

           0            0.95      1.00      0.97       929
           1            0.00      0.00      0.00        53
          accuracy                          0.95       982
          macro avg     0.47      0.50      0.49       982
       weighted avg     0.89      0.95      0.92       982



* Berikut merupakan hasil dari confusion matrix pada model Random Forest:

![Hasil CM Rf](https://github.com/iqkyu5566/strokeprediction/blob/main/img/datapreparation1.png?raw=true)

* Evaluasi Hasil Model Random Forest dengan Hyperparameter GridSearch:

              precision    recall  f1-score   support

           0              0.95      1.00      0.97       929
           1              0.00      0.00      0.00        53

           accuracy                           0.95       982
          macro avg       0.47      0.50      0.49       982
       weighted avg       0.89      0.95      0.92       982

* Berikut merupakan hasil dari confusion matrix pada model Random Forest dengan Hyperparameter GridSearch:

![CM RF_Tuning](https://github.com/iqkyu5566/strokeprediction/blob/main/img/datapreparation2.png?raw=true)

* Evaluasi Hasil Model Support Vektor Machine:

Akurasi: 0.9460285132382892

           precision    recall  f1-score   support
           0             0.95      1.00      0.97       929
           1             0.00      0.00      0.00        53

         accuracy                            0.95       982
         macro avg       0.47      0.50      0.49       982
         weighted avg    0.89      0.95      0.92       982


* Berikut merupakan hasil dari confusion matrix pada model Support Vektor Machine:

![CM SVM](https://github.com/iqkyu5566/strokeprediction/blob/main/img/datapreparation3.png?raw=true)


## Kesimpulan

Berdasarkan hasil pelatihan model dengan dua algoritma machine learning, yaitu Random Forest dan Support Vector Machine (SVM), masing-masing model menunjukkan kemampuan yang baik dalam memprediksi risiko stroke. Berikut adalah ringkasan hasil yang diperoleh:

Model SVM:

Sebelum optimasi, model SVM menghasilkan akurasi sebesar 94.6%.
Setelah dilakukan optimasi menggunakan Hyperparameter GridSearch, akurasi meningkat menjadi 94.8%.
Model ini menunjukkan performa terbaik dalam memisahkan data berdasarkan risiko stroke.
Model Random Forest:

Sebelum optimasi, akurasi yang dihasilkan adalah 95%.
Setelah optimasi, akurasi tetap di 95%, namun masih mengalami kesulitan dalam memprediksi kelas minoritas (pasien dengan stroke).
Model SVM dengan optimasi terbukti menjadi algoritma terbaik untuk dataset ini karena memberikan performa yang lebih konsisten meskipun data target tidak seimbang. Namun, kedua model memiliki tantangan dalam memprediksi kelas minoritas (stroke) akibat distribusi data yang tidak seimbang.

Rekomendasi:
Gunakan teknik seperti oversampling (SMOTE) atau weight adjustment untuk meningkatkan kemampuan model dalam memprediksi kelas minoritas.
Perlu dilakukan eksperimen lebih lanjut dengan dataset yang lebih besar dan beragam untuk meningkatkan generalisasi model.

## Penutup

Proyek ini menunjukkan bahwa teknologi machine learning dapat digunakan secara efektif untuk membantu mendeteksi risiko stroke, terutama dalam situasi di mana diagnosis dini sangat penting untuk mengurangi angka kematian dan kecacatan. Dengan memanfaatkan algoritma seperti Random Forest dan SVM, serta melakukan optimasi hyperparameter, proyek ini berhasil membangun model prediktif yang akurat.

Hasil dari proyek ini diharapkan dapat menjadi langkah awal dalam mendukung profesional kesehatan untuk mengidentifikasi pasien yang memiliki risiko tinggi terkena stroke. Meskipun ada tantangan, solusi ini memberikan panduan yang jelas untuk pengembangan sistem prediksi berbasis data di masa depan.

Terima kasih telah membaca laporan ini. Semoga laporan ini dapat memberikan manfaat bagi perkembangan teknologi dan pengaplikasiannya dalam dunia kesehatan.

