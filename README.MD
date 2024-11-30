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

| No  | Variabel          | Deskripsi                                                              |
| --- | ----------------- | ---------------------------------------------------------------------- |
| 1   | gender            | Jenis kelamin pasien (Male/Female/Other)                               |
| 2   | age               | Usia pasien                                                            |
| 3   | hypertension      | Riwayat hipertensi (1: Ya, 0: Tidak)                                   |
| 4   | heart_disease     | Riwayat penyakit jantung (1: Ya, 0: Tidak)                             |
| 5   | ever_married      | Status pernikahan pasien (Yes/No)                                      |
| 6   | work_type         | Jenis pekerjaan (Private/Self-employed/Govt_job/Children/Never_worked) |
| 7   | Residence_type    | Tipe tempat tinggal (Urban/Rural)                                      |
| 8   | avg_glucose_level | Rata-rata tingkat glukosa dalam darah                                  |
| 9   | bmi               | Indeks Massa Tubuh                                                     |
| 10  | smoking_status    | Status merokok (never smoked/formerly smoked/smokes/unknown)           |
| 11  | stroke            | Target variabel (1: Mengalami stroke, 0: Tidak mengalami stroke)       |

### Informasi Data

Jumlah Data: 5110 entri
Fitur: 11 kolom
Variabel Target: stroke
Missing Values: Terdapat nilai hilang pada kolom bmi.

Memvisualisasikan data menggunakan boxplot untuk fitur numerik: [age,hypertension, heart_disease, avg_glucose_level, bmi]

![Visualisasi Boxplot Age](https://github.com/iqkyu5566/strokeprediction/blob/main/img/datapreparation1.png?raw=true)

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

![Confusion Matrix Rf](https://github.com/user-attachments/assets/68c06849-f6a7-4d51-ade6-2e1429c8c0a3)

- Hyperparameter Model Random Forest dengan GridSearch

Parameter yang digunakan untuk optimasi model random forest menggunakan GridSearch yaitu:

- 'n_estimators': [50, 100, 200]
- 'max_depth': [None, 10, 20, 30]
- 'min_samples_split': [2, 5, 10]

Dari parameter diatas akan dicari nilai parameter terbaik menggunakan GridSearch untuk model klasifikasi random forest.

> Hasil parameter terbaik dari Hyperparameter GridSearch yaitu:

> - 'max_depth': 20
> - 'min_samples_split': 5
> - 'n_estimators': 200

> Berikut penjelasan dari proses Hyperparamer Tuning dan GridSearch terhadap model:
>
> - Hyperparameter tuning dapat digunakan untuk memastikan performa terbaik dari model yang diterapkan, kita akan menggunakan Hyperparameter Tuning dengan GridSearch. Baik Random Forest maupun SVM memiliki hyperparameter yang dapat mempengaruhi performa model secara signifikan. Misalnya, pada Random Forest, jumlah tree (n_estimators) atau kedalaman maksimum tree (max_depth) perlu dioptimalkan, sedangkan pada SVM, parameter seperti C (regularization) dan kernel (linear, polynomial, atau RBF) harus disesuaikan.
> - GridSearch adalah metode yang memungkinkan kita untuk menguji kombinasi berbagai nilai hyperparameter dan memilih yang terbaik berdasarkan kinerja model. Dalam proyek ini, GridSearch akan menguji berbagai kombinasi parameter dan mengevaluasi model berdasarkan metrik seperti akurasi, precision, recall, dan F1-score. Dengan melakukan tuning yang tepat, model dapat dioptimalkan untuk memberikan hasil klasifikasi yang lebih baik dan akurat dalam memprediksi kisaran harga ponsel. Keunggulan Hyperparameter Tuning dengan GridSearch adalah dapat meningkatkan performa model, dengan menemukan kombinasi hyperparameter terbaik, model akan bekerja lebih optimal dan memberikan hasil klasifikasi yang lebih akurat. Serta keunngulan lainnya dapat mencegah overfitting, dengan pengaturan hyperparameter yang tepat, kita dapat menghindari overfitting dan memastikan bahwa model dapat bekerja dengan baik pada data baru.
>   Setelah dilakukan optimasi, model yang terbaik akan dievaluasi menggunakan metrik seperti akurasi, precision, recall, dan F1-score, untuk memastikan bahwa prediksi kisaran harga yang dihasilkan dapat diimplementasikan secara efektif dalam pengambilan keputusan penetapan harga perusahaan.

- Model prediksi dengan algoritma Support Vektor Machine:

```
# Inisialisasi model SVM
model_svm = SVC(kernel='rbf', random_state=42)
model_svm.fit(X_train, y_train)
```

> Support Vector Machine (SVM) adalah algoritma yang bekerja dengan menemukan hyperplane terbaik yang memisahkan kelas-kelas data. Dalam proyek ini, SVM akan digunakan untuk memetakan data fitur ponsel ke dalam ruang berdimensi tinggi, kemudian menemukan garis atau kurva (hyperplane) yang memisahkan ponsel berdasarkan kisaran harga. SVM sangat efektif ketika ada perbedaan yang jelas antara kategori harga, dan algoritma ini mampu bekerja dengan baik bahkan ketika data tidak linear, melalui penggunaan kernel trick. Keunggulan SVM yaitu memiliki keakuratan pada data yang tidak seimbang SVM mampu memberikan hasil klasifikasi yang baik, bahkan dalam kasus di mana data tidak seimbang atau memiliki sedikit kesalahan klasifikasi. Serta keunggulan lainnya yaitu dalam penggunaan kernel trick dimana SVM dapat menangani data yang tidak linear dan membuat model lebih fleksibel untuk berbagai macam distribusi data fitur ponsel.

> Berikut merupakan penjelasan terhadap setiap parameter yang digunakan:

> - kernel = Algoritma SVM menggunakan serangkaian fungsi matematika yang didefinisikan sebagai kernel. Fungsi kernel adalah mengambil data sebagai input dan mengubahnya ke dalam bentuk yang dibutuhkan. Algoritma SVM yang berbeda menggunakan berbagai jenis fungsi kernel. Fungsi-fungsi ini dapat memiliki tipe yang berbeda. Misalnya linear, nonlinier, polinomial, fungsi basis radial (RBF), dan sigmoid.
> - random_state = mengontrol random number generator yang digunakan. Parameter ini berupa bilangan integer dan nilainya bebas. Parameter ini bertujuan untuk memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan. Jika tidak ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan tes berbeda. Hal ini berpengaruh terhadap akurasi model ML yang menjadi berbeda tiap kali di-run.

- Confusion Matrix model Support Vektor Machine (SVM):

  > Confusion Matrix digunakan untuk melihat hasil prediksi dari model SVM

- Model optimasi algoritma Support Vektor Machine (SVM) dengan Hyperparameter GridSearch:

Parameter yang digunakan untuk optimasi model SVM menggunakan GridSearch yaitu:

- 'C': [0.1, 1, 10, 100]
- 'gamma': [1, 0.1, 0.01, 0.001]
- 'kernel': ['rbf', 'poly', 'sigmoid']

dari parameter diatas akan dicari nilai parameter terbaik menggunakan GridSearch untuk model klasifikasi SVM.
kemudian akan dilihat kembali confusion matrix setelah optimasi.

> Hasil parameter terbaik dari Hyperparameter GridSearch yaitu:

> - 'C': 0.1
> - 'gamma': 1
> - 'kernel': 'poly'

## Evaluasi

Untuk melihat hasil pelatihan dari masing-masing model klasifikasi dengan menggunakan akurasi pada nilai yang dihasilkan pada setiap model, nilai akurasi menggunakan library dari [sklearn](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html). Selainmelihat nilai akurasi pada proyek ini melakukan visualisasi hasil pelatihan dengan confusion matrix. Berikut merupakan hasil akurasi pada setiap model:

- Evaluasi Hasil Model Random Forest:

![Akurasi RF](https://github.com/user-attachments/assets/e8c08739-c5b5-49d0-8980-d23ce916a524)

- Berikut merupakan hasil dari confusion matrix pada model Random Forest:

![Hasil CM Rf](https://github.com/user-attachments/assets/2186566e-5190-429b-95be-d7280b4e0c9c)

- Evaluasi Hasil Model Random Forest dengan Hyperparameter GridSearch:

![Akurasi RF_Tuning](https://github.com/user-attachments/assets/99b484a1-8774-427f-b16c-687a270a2661)

- Berikut merupakan hasil dari confusion matrix pada model Random Forest dengan Hyperparameter GridSearch:

![CM RF_Tuning](https://github.com/user-attachments/assets/d4dcd251-35e8-4cc3-96e8-e543a979031a)

- Evaluasi Hasil Model Support Vektor Machine:

![Akurasi SVM](https://github.com/user-attachments/assets/8a91ca7c-8a41-4920-88c3-587682bc9761)

- Berikut merupakan hasil dari confusion matrix pada model Support Vektor Machine:

![CM SVM](https://github.com/user-attachments/assets/71614ca3-c1ba-4dd5-b9e0-2fa922088575)

- Evaluasi Hasil Model SVM dengan Hyperparameter GridSearch:

![Akurasi SVM_Tuning](https://github.com/user-attachments/assets/644ede6c-7d3d-4446-b31f-feef2b47c8a3)

- Berikut merupakan hasil dari confusion matrix pada model SVM dengan Hyperparameter GridSearch:

![CM SVM_Tuning](https://github.com/user-attachments/assets/f3b17eae-2299-484b-b9b5-891cd6aca847)

- Membuat Plot metrik Akurasi dengan bar chart:

![Perbandingan Model](https://github.com/user-attachments/assets/8ccaebae-4de2-4ae9-bda0-bd12ca124b6e)

- Pengujian model prediksi dengan model SVM menggunakan data test yang sudah disediakan dengan data tanpa ada variabel price_range. Berikut hasil prediksi kemudian tersimpan dalam drive dengan file csv, berikut tabel hasil prediksi data test menggunakan model SVM:

![Prediksi dataset](https://github.com/user-attachments/assets/29d68c34-0fd4-4121-be3b-8275304e7f98)

## Kesimpulan

Berdasarkan hasil pelatihan model dengan dua algoritma machine learning yaitu Random Forest dan Support Vektor Machine, masing-masing model mampu memprediksi diatas 85% dan mengalami peningkatan nilai akurasi saat dilakukan optimasi menggunakan Hyperparameter GridSearch, untuk akurasi tertinggi dihasilkan oleh model SVM dengan akurasi sebesar 96% sebelum optimasi dan 97% sesudah dilakukan optimasi, sementara untuk algoritma Random Forest memiliki nilai akurasi sebesar 89% dengan peningkatan yang cukup tinggi dibandingkan dengan SVM setelah dilakukan optimasi yaitu sebesar 3% dengan akurasi menjadi 91%. Dari hasil tersebut model algoritma Support Vektor Machine (SVM) lebih tepat dan baik untuk digunakan sebagai model klasifikasi pada dataset Mobile Price Kalsifikasi.

## Penutup

Demikian hasil dari laporan proyek machine learning, predicitive analytics dengan dataset Mobile Price Klasifikasi. Bilamana didalam penyampaian serta penjelasan yang kurang berkenaan, saya memohon maaf. Atas waktu dan perhatiannya, saya ucapkan Terima kasih telah membaca laporan ini. Semoga dapat memberi manfaat bagi kita semuanya.
