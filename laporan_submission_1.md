# Laporan Proyek Machine Learning - Dwi Sandi Kalla

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang menjadi masalah kesehatan global. Diabetes terjadi ketika pankreas tidak memproduksi insulin dalam jumlah cukup atau ketika tubuh tidak dapat menggunakan insulin yang diproduksi dengan baik. Menurut laporan dari _World Health Organization_ (WHO), pada tahun 2021 diabetes merupakan penyebab langsung dari 1,6 juta kematian dan 47% dari semua kematian akibat diabetes dialami oleh orang sebelum berusia 70 tahun.[[1]](https://www.who.int/news-room/fact-sheets/detail/diabetes). Seiring waktu, diabetes dapat merusak sistem pembuluh darah di jantung, mata, ginjal, dan saraf. Diabetes dapat menyebabkan hilangnya penglihatan secara permanen karena rusaknya pembuluh darah di mata. Banyak penderita diabetes mengalami masalah pada kaki karena kerusakan saraf dan aliran darah yang cenderung buruk. Hal ini menyebabkan terjadinya borok kaki dan dapat berujung pada amputasi.

Deteksi dini terhadap risiko diabetes dapat menjadi langkah yang sangat awal untuk mengurangi penyakit lain yang disebabkan oleh diabetes. Dalam hal ini, pendekatan berbasis data dan _Artificial Intelligence_ (AI) menjadi salah satu solusi yang dapat diandalkan. Perkembangan teknologi machine learning dan data science telah membuka peluang baru untuk menganalisis data kesehatan dalam skala besar dan menghasilkan model prediksi yang akurat.

Proyek ini bertujuan untuk membangun model prediksi diabetes dengan memanfaatkan Diabetes Prediction Dataset yang saya peroleh dari kaggle dan berisi berbagai indikator kesehatan seperti jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, indeks massa tubuh (_Body Mass Index_), umur, serta faktor genetik melalui _Diabetes Pedigree Function_. Dataset ini disusun untuk mendukung pengembangan model prediktif yang mampu mengidentifikasi individu dengan risiko tinggi terkena diabetes secara lebih akurat dan efisien.

Dengan menerapkan algoritma machine learning seperti K-Nearest Neighbors, Random Forest, Linear Regression, proyek ini akan mengevaluasi performa masing-masing model berdasarkan metrik _Mean Squared Error_ (MSE). Proyek ini diharapkan tidak hanya menghasilkan model prediktif yang efektif, namun juga dapat memberikan wawasan bagi praktisi kesehatan dalam pengambilan keputusan yang berbasis data.

Penerapan sistem berbasis AI dalam prediksi diabetes mampu meningkatkan akurasi diagnosis hingga 87% dibanding metode konvensional [[2]](https://www.jmir.org/2018/5/e10775/). Hal ini menunjukkan potensi besar dari pendekatan teknologi dalam meningkatkan layanan kesehatan preventif. Dengan latar belakang tersebut, pengembangan model prediksi diabetes berbasis machine learning menjadi suatu kebutuhan penting dalam upaya deteksi dini dan pengendalian penyakit diabetes, khususnya dalam masyarakat yang belum memiliki akses terhadap pemeriksaan kesehatan rutin.

## Business Understanding

### Problem Statements

- Bagaimana proses pembersihan data seperti penanganan missing value, outlier, dan duplikat dapat meningkatkan kualitas dataset serta membantu menghasilkan model prediksi diabetes yang lebih akurat dan andal?
- Bagaimana membangun model prediksi untuk mengidentifikasi risiko seseorang terkena diabetes berdasarkan data jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, indeks massa tubuh (Body Mass Index), umur, serta faktor genetik melalui Diabetes Pedigree Function?
- Algoritma machine learning mana yang memberikan performa terbaik berdasarkan nilai MSE dalam memprediksi diabetes pada dataset yang digunakan?

### Goals

- Melakukan proses pembersihan data melalui identifikasi dan penanganan missing values, outlier, serta data duplikat guna meningkatkan kualitas data input, sehingga model machine learning dapat belajar dari informasi yang bersih dan representatif untuk memprediksi risiko diabetes.
- Mengembangkan model prediktif berbasis data kesehatan untuk mengidentifikasi individu dengan risiko diabetes secara akurat.
- Membandingkan performa beberapa algoritma machine learning seperti K-Nearest Neighbors, Random Forest, dan Linear Regression menggunakan metrik Mean Squared Error (MSE).

### Solution Statement

- Menerapkan beberapa algoritma machine learning seperti K-Nearest Neighbors (KNN), Random Forest, dan Linear Regression untuk membangun model prediksi diabetes, dengan melakukan pembandingan performa menggunakan metrik evaluasi _Mean Squared Error_ (MSE) pada data latih dan uji
- Melakukan pembersihan data secara menyeluruh sebelum pelatihan model, mencakup penanganan missing values, penghapusan duplikat, dan deteksi outlier menggunakan teknik seperti IQR (Interquartile Range).

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [_Healthcare Diabetes Dataset_](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes) yang berasal dari kaggle yang dirancang untuk keperluan prediksi risiko diabetes berdasarkan beberapa faktor. Dataset ini terdiri dari 2768 baris dan 10 kolom, dimana setiap baris merepresentasikan satu individu. Data ini memiliki satu variabel target yang bersifat biner, yaitu "Outcome" yang menunjukkan seseorang mengidap diabetes (1) atau tidak (0).

### Variabel-variabel pada _Healthcare Diabetes Dataset_ adalah sebagai berikut:
- Id : 	Pengidentifikasi unik untuk setiap baris data atau individu. Kolom ini bersifat administratif dan tidak memiliki peran langsung dalam proses prediksi. Biasanya akan dihapus pada tahap preprocessing.
- _Pregnancies_ (Kehamilan) : Menunjukkan jumlah kehamilan yang pernah dialami oleh pasien wanita.
- _Glucose_ (Glukosa) : Menggambarkan kadar glukosa plasma dalam darah setelah 2 jam dalam uji toleransi glukosa oral.
- _BloodPressure_ (Tekanan Darah) : Merujuk pada tekanan darah diastolik, yaitu tekanan saat jantung berada dalam keadaan rileks di antara dua detakan. Tekanan darah tinggi sering ditemukan bersamaan dengan diabetes dan termasuk dalam sindrom metabolik.
- _SkinThickness_ (Ketebalan Kulit) : Mengukur ketebalan lipatan kulit triceps, yang dapat digunakan sebagai indikator lemak tubuh.
- Insulin : Menunjukkan kadar insulin serum 2 jam setelah tes toleransi glukosa. Kadar insulin dapat membantu menilai resistensi insulin yang merupakan akar penyebab utama diabetes.
- BMI : Singkatan dari _Body Mass Index_, yaitu rasio berat badan terhadap tinggi badan dalam meter kuadrat. BMI merupakan indikator yang umum digunakan untuk mengklasifikasikan individu sebagai kurus, normal, gemuk, atau obesitas — faktor penting dalam prediksi diabetes.
- _DiabetesPedigreeFunction_ : Skor yang mengindikasikan faktor keturunan/genetik risiko diabetes. Nilai ini dihitung berdasarkan riwayat diabetes dalam keluarga dan kompleksitas hubungan kekerabatan.
- _Age_ (Usia) : Usia pasien dalam tahun.
- _Outcome_ (Hasil) : **1** Individu tersebut menderita diabetes, **2** Individu tersebut tidak menderita diabetes.

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) adalah proses memahami struktur, pola, dan anomali dari sebuah dataset sebelum dilakukan pemodelan. Pada proyek ini dilakukan beberapa tahapan EDA sebagai berikut:
1. Pemeriksaan Awal pada Data
   - Memeriksa fitur data dengan `data.info()`
     <br>Digunakan untuk melihat struktur dataset, termasuk jumlah baris dan kolom, tipe data, dan apakah ada nilai yang hilang. Hasilnya menunjukkan bahwa semua kolom memiliki tipe data numerik dan tidak terdapat missing values (kosong) dengan total kolom 9 dan baris 2768.
   - Mengecek duplikasi data `data.duplicated().sum()`
     <br>Mengecek apakah ada data yang duplikat atau tidak. Untuk hasil pada pengecekan duplikasi, data tidak memiliki nilai duplikat.
   - Memastikan tidak ada missing value `data.isna().sum()`
     <br>Memastikan bahwa tidak ada nilai _NaN_ untuk setiap kolom yang menandakan missing value.
   - Penghapusan kolom **Id**
     <br>Kolom **Id** dihapus karena tidak relevan terhadap proses prediksi, hanya berfungsi sebagai identifikasi unik.
2. Statistika Deskriptif
3. Deteksi Outlier dengan Boxplot
5. Penghapusan Outlier dengan IQR Method
6. Distribusi Fitur (Histogram)
7. Pairplot (Visualisasi Hubungan antar Variabel)
8. Heatmap Korelasi

## Data Preparation
Tahap Data Preparation merupakan langkah penting sebelum melakukan proses training model machine learning. Tujuannya adalah untuk memastikan data dalam kondisi optimal agar model dapat belajar secara efektif. Berikut ini adalah tahapan data preparation yang dilakukan dalam proyek ini:
1. Mengecek Ringkasan Informasi Dataset
   - Mengecek informasi data menggunakan `data.info()`.
   - Tujuan dari pengecekan ini adalah untuk membantu memahami struktur data, Mengidentifikasi nilai yang hilang, memeriksa setiap type data dari setiap kolom, dan juga termasuk langkah awal untuk proses _data cleaning_.

2. Mengecek Duplikasi Data
   - Mengecek duplikasi data dilakukan dengan kode `data.duplicated().sum()`.
   - Pada proyek ini tidak ditemukan adanya data yang duplikat.
   - Mengecek duplikasi data bertujuan agar data tidak ganda, data ganda dapat mendominasi hasil perhitungan statistik yang menghasilkan kesimpulan yang bias. Proses pengecekan duplikasi diperlukan untuk mendapatkan representasi data yang akurat, efisien, dan relevan untuk pengambilan keputusan yang tepat.

3. Pemeriksaan dan Penanganan Nilai Kosong (Missing Values)
   - Mengecek missing value dapat menggunakan `data.isna().sum()`
   - Pada proyek ini tidak ditemukan adanya missing value.
   - Nilai kosong dapat mengganggu proses pelatihan model. Jika ada, harus dilakukan penanganan seperti imputasi (pengisian nilai) atau penghapusan baris/kolom.

4. Penghapusan Kolom **Id**
   - Penghapusan kolom **Id** dilakukan melalui kode berikut `data.drop('Id', axis=1, inplace=True)`.
   - Kolom **Id** hanya berisi identitas pasien dan tidak memiliki kontribusi terhadap prediksi diabetes. Kolom seperti ini disebut irrelevant feature dan dapat menyebabkan noise dalam proses pelatihan model.

5. Pengecekan Outlier dengan Box-plot
   - Pengecekan outlier menggunakan boxplot untuk setiap kolom dilakukan dengan menggunakan
     ```python
     for column in data.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot for {column}')
        plt.show()
     ```
   - Hasilnya setiap kolom memiliki nilai outlier kecuali kolom Outcome.
   -  Beberapa kolom/fitur memiliki nilai outlier yang jika tidak ditangani, outlier bisa menyebabkan model belajar pola yang tidak benar (overfitting atau bias).
   -  > Gambar dapat dilihat di : [Google Collabs - project](https://colab.research.google.com/drive/1XQ_spIaupa-1KupVIS4ozF7IhCmrcStA?usp=sharing)

6. Menangani Outlier (IQR)
   - Penanganan dilakukan dengan metode Interquartile Range (IQR) yang dilakukan memalui kode berikut:
     ```python
     Q1 = data.quantile(0.25)
     Q3 = data.quantile(0.75)
     IQR = Q3 - Q1

     # Menghapus baris yang mengandung outlier
     data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
     data.info()
     ```
   - Setelah penghapusan outlier jumlah baris yang semula 2768 menjadi 2299 baris.
   - Alasan dilakukan penerapan metode IQR adalah karena ingin menghapus outlier agar nantinya tidak berpengaruh ke model.

7. Distribusi Fitur (Histogram)
   - Disitribusi fitur dilakukan melalui pembuatan histogram dengan kode berikut :
     ```python
     data.hist(bins=50, figsize=(20,15))
     plt.show()
     ```
   - Hasil dari distribusi nya adalah sebagai berikut:
     * Pregnancies, Age, DiabetesPedigreeFunction, SkinThickness, Insulin: Distribusinya miring ke kanan (right-skewed).
     * Glucose & BMI: Hampir normal, sedikit miring ke kanan.
     * BloodPressure: Simetris, mendekati normal.
     * Outcome: Data biner dan tidak seimbang (lebih banyak kelas 0).
    - Alasan dilakukannya ini adalah untuk mencari temuan baru terkait data.

8. Korelasi Antar Fitur
   - Menggunakan correlation matrix dan pairplot
     ```python
     # pairplot
     sns.pairplot(data, diag_kind = 'kde')
     # correlation matrix
     plt.figure(figsize=(10, 8))
     correlation_matrix = data.corr().round(2)
     sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
     plt.title("Correlation Matrix", size=20)
     ```
   - Hasil dari correlation matrix adalah
     * Pregnancies: Korelasi 0.23, arah korelasi positif (semakin banyak kehamilan, sedikit cenderung lebih tinggi risiko diabetes).
     * Glucose: Korelasi 0.5, arah korelasi positif (semakin tinggi kadar glukosa, semakin tinggi risiko diabetes).
     * BloodPressure: Korelasi 0.18, arah korelasi positif (semakin tinggi tekanan darah, sedikit cenderung lebih tinggi risiko diabetes).
     * SkinThickness: Korelasi 0.04, arah korelasi positif (semakin tebal lipatan kulit, sangat sedikit cenderung lebih tinggi risiko diabetes).
     * Insulin: Korelasi 0.11, arah korelasi positif (semakin tinggi kadar insulin, sedikit cenderung lebih tinggi risiko diabetes).
     * BMI: Korelasi 0.25, arah korelasi positif (semakin tinggi BMI, sedikit cenderung lebih tinggi risiko diabetes).
     * DiabetesPedigreeFunction: Korelasi 0.16, arah korelasi positif (semakin tinggi fungsi silsilah diabetes, sedikit cenderung lebih tinggi risiko diabetes).
     * Age: Korelasi 0.29, arah korelasi positif (semakin bertambah usia, sedikit cenderung lebih tinggi risiko diabetes).
   - Analisis korelasi digunakan untuk menentukan fitur yang relevan dalam pemodelan, di mana fitur dengan korelasi tinggi terhadap variabel target (seperti Outcome) layak dipertahankan karena berkontribusi signifikan. Selain itu, korelasi juga membantu menghindari multikolinearitas, yaitu kondisi ketika dua fitur memiliki hubungan yang sangat kuat (misalnya antara SkinThickness dan Insulin), yang dapat mengganggu stabilitas model. Meskipun begitu, fitur dengan korelasi rendah bukan berarti tidak penting, karena kombinasi antar fitur tetap dapat meningkatkan performa model secara keseluruhan.

9. Splitting Data
    - Kode yang diterapkan dalam splitting data adalah sebagai berikut:
      ```python
      X = data.drop(["Outcome"],axis =1)
      y = data["Outcome"]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
      ```
    - Data terbagi menjadi 20% untuk `X_test` dan `y_test` serta 80% untuk `X_train` dan `y_train`. Melakukan stratified splitting yaitu dengan `stratify=y`.
    - Data train digunakan untuk melatih model, data test digunakan untuk menguji generalisasi model terhadap data baru.

10. Standarisasi Data training dan testing
    - Proses standarisasi mengubah data agar memiliki rata-rata 0 dan standar deviasi 1, sehingga setiap fitur berkontribusi secara seimbang. Penting untuk fit hanya pada data training, lalu transformasi yang sama digunakan pada data testing, agar tidak terjadi kebocoran informasi dari data testing ke model (data leakage) dan hasil evaluasi tetap valid.
    - Standarisasi data diperlukan karena banyak algoritma machine learning, seperti K-Nearest Neighbors, Support Vector Machine, dan Logistic Regression, sensitif terhadap skala fitur. Jika fitur memiliki skala yang berbeda (misalnya, satu fitur dalam satuan puluhan dan fitur lain dalam ratusan), maka algoritma bisa lebih "memperhatikan" fitur dengan skala besar, sehingga menghasilkan model yang tidak akurat.
    - Untuk melakukan standarisasi dugunakan kode sebagai berikut:
      ```python
      # Standarisasi untuk Training
      scaler = StandardScaler()
      scaler.fit(X_train)
      X_train = scaler.transform(X_train)
      X_train = pd.DataFrame(X_train, columns=X.columns)
      X_train
      # Stanndarisasi untuk testing
      X_test = scaler.transform(X_test)
      X_test = pd.DataFrame(X_test, columns=X.columns)
      X_test
      ```

## Modeling
Proyek ini menggunakan tiga model machine learning, yaitu K-Nearest Neighbors (KNN), Random Forest, Linear Regression. Ketiga model ini dilatih dengan menggunakan data yang telah melalui tahap preprocessing, serta dievaluasi menggunakan metrik Mean Squared Error (MSE). Berikut penjabaran ketiga metode tersebut:
1. K-Nearest Neighbors (KNN)
   - Parameter yang digunakan : `n_neighbors = 2`, ini berarti prediksi didasarkan pada rata-rata tetangga terdekat.
   - Cara Kerja : KNN melakukan prediksi berdasarkan kedekatan data (jarak Euclidean) dengan tetangga terdekatnya di data pelatihan. KNN sangat tergantung pada kualitas dan distribusi data.
   - Kelebihan : Non-parametrik, sederhana dan mudah dimplementasikan, dan bisa menangkap pola lokal dengan baik.
   - Kekurangan : Sensitif terhadap pola skala data, Lambat untuk dataset besar karena perhitungan jarak terhadap semua titik data pelatihan, Rentan terhadap noise dan outlier.
     ```python
     knn = KNeighborsRegressor(n_neighbors=2)
     knn.fit(X_train, y_train)
     models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
     ```

2. Random Forest
   - Parameter yang digunakan :
     * `n_estimators = 200` : jumlah pohon dalam hutan.
     * `max_depth = 20` : kedalaman maksimum pohon.
     * `min_samples_split = 2` : jumlah minimum sampel untuk membagi node.
     * `random_state = 42` : untuk menjaga hasil tetap konsisten.
   - Cara Kerja : Random Forest membentuk banyak pohon keputusan (decision tree) dan menggabungkan hasilnya (rata-rata untuk regresi) agar lebih stabil dan akurat. Random Forest juga menggunakan subset fitur dan data (bagging) untuk membangun tiap pohon.
   - Kelebihan : Mampu menangkap hubungan non-linier, tidak sensitif terhadap outlier dan multikolinearitas, bias rendah dan akurasi tinggi.
   - Kekurangan : Waktu komputasi bisa tinggi, apalagi jika pohon sangat dalam, lebih sulit diinterpretasi dibanding regresi linear.
     ```python
     rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, min_samples_split=2)
     rf.fit(X_train, y_train)
     models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=rf.predict(X_train), y_true=y_train)
     ```

3. Linear Regression
   - Parameter yang digunakan : `sklearn.linear_model.LinearRegression()`.
   - Cara Kerja : Linear regression mencari garis lurus terbaik yang meminimalkan jumlah kuadrat kesalahan antara prediksi dan nilai sebenarnya. Model ini mengasumsikan hubungan linier antara fitur dan target.
   - Kelebihan : Mudah diinterpretasikan, membutuhkan waktu komputasi yang cepat.
   - Kekurangan : Tidak mampu menangkap hubungan non-linear, sensitif terhadap multikolinearitas, Asumsi normalitas dan homoskedastisitas sering tidak terpenuhi dalam data nyata.
     ```python
     lr = LinearRegression()
     lr.fit(X_train, y_train)
     models.loc['train_mse','LinearRegression'] = mean_squared_error(y_pred=lr.predict(X_train), y_true=y_train)
     ```
Model terbaik yang dipilih adalah Random Forest, karena menghasilkan nilai Mean Squared Error (MSE) terkecil pada data uji, yaitu sebesar 0.000007, dibandingkan dengan model KNN dan Linear Regression. Selain itu, model ini juga menunjukkan performa yang konsisten antara data latih dan data uji, menandakan kemampuan generalisasi yang baik tanpa overfitting. Random Forest juga unggul dalam menangkap hubungan non-linier antar fitur, sehingga lebih efektif dalam menyelesaikan permasalahan regresi pada dataset ini.Hasil algoritma yang terbaik berdasarkan metrik yang diperoleh.

## Evaluation
Pada proyek ini digunakan metrik Mean Squared Error (MSE). MSE adalah salah satu metrik evaluasi yang paling umum digunakan dalam masalah regresi. Metrik ini bekerja dengan mengukur rata-rata selisih antara nilai aktual dan nilai prediksi dari suatu model. Dengan kata lain, MSE memberitahu bahwa seberapa jauh model dari kenyataan sebenarnya dalam satuan kuadrat. MSE digunakan karena sensitif terhadap error besar, mudah dihitung dan dibedakan antar model, dan konsisten secara matematis. Rumus yang digunakan dalam MSE adalah

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

dengan : <br>
${n}$ : jumlah data<br>
$y_i$ : nilai aktual<br>
$\hat{y}_i$ : nilai prediksi<br>
$(y_i - \hat{y}_i)^2$ : selisih kuadrat antara nilai aktual dan prediksi<br>

Dari ketiga model, didapatkan nilai MSE adalah sebagai berikut: 

| Model | train | test |
| --- | --- | --- |
| KNN | 0.000002 | 0.000009 |
| Random Forest | 0.000002 | 0.000007 |
| Linear Regression | 0.000144 | 0.000147 |

Berdasarkan hasil evaluasi menggunakan metrik Mean Squared Error (MSE) pada data train dan test, diperoleh bahwa model Random Forest memiliki performa terbaik dibandingkan model lainnya. Hal ini ditunjukkan oleh nilai MSE yang paling kecil, yaitu 0.000002 pada data train dan 0.000007 pada data test. Artinya, model Random Forest mampu memprediksi target dengan kesalahan yang sangat kecil.

Model KNN juga menunjukkan performa yang baik dengan MSE train 0.000002 dan test 0.000009, meskipun sedikit lebih besar dibandingkan Random Forest.

Sementara itu, model Linear Regression menunjukkan performa yang paling rendah di antara ketiganya, dengan MSE train 0.000144 dan test 0.000147. Ini menunjukkan bahwa model tersebut tidak mampu menangkap kompleksitas data sebaik dua model lainnya.


## Daftar Referensi
> [1]World Health Organization, Diabetes, Nov. 14, 2024. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/diabetes
>
> [2] I. Contreras and J. Vehi, "Artificial intelligence for diabetes management and decision support: literature review," Journal of Medical Internet Research, vol. 20, no. 5, p. e10775, 2018.doi: e10775.https://www.jmir.org/2018/5/e10775/

