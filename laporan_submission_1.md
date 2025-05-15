# Laporan Proyek Machine Learning - Dwi Sandi Kalla

## Domain Proyek

Gaya hidup sedentari dan kurangnya aktivitas fisik merupakan salah satu faktor risiko utama dari berbagai penyakit tidak menular, termasuk obesitas, penyakit jantung, dan diabetes. Menurut World Health Organization (WHO), kurangnya aktivitas fisik menjadi penyebab sekitar 2 juta kematian setiap tahunnya di seluruh dunia dan merupakan faktor risiko keempat utama penyebab kematian global [1]. Selain itu, lebih dari 1,4 miliar orang dewasa tidak cukup aktif secara fisik, yang meningkatkan risiko morbiditas dan mortalitas secara signifikan.[[1]](https://www.who.int/news-room/fact-sheets/detail/physical-activity). Salah satu cara untuk mendorong gaya hidup aktif adalah dengan memantau jumlah kalori yang terbakar selama aktivitas fisik. Pemantauan ini tidak hanya bermanfaat bagi individu yang ingin menurunkan berat badan atau peningkatan massa otot, tetapi juga untuk mempertahankan kebugaran dan kesehatan metabolik secara keseluruhan.

Namun, perhitungan kalori yang terbakar secara manual sering kali tidak akurat karena tidak mempertimbangkan faktor individual seperti usia, jenis kelamin, berat badan, suhu tubuh, dan detak jantung. Dalam hal ini, pendekatan berbasis machine learning (ML) mampu menawarkan prediksi yang lebih akurat dan personal dengan memanfaatkan data fisiologis dan karakteristik aktivitas pengguna. Dengan meningkatnya kesadaran akan gaya hidup sehat, prediksi kalori terbakar menjadi aspek penting dalam aplikasi kebugaran modern.

Perhitungan jumlah kalori yang terbakar selama aktivitas fisik umumnya mempertimbangkan faktor-faktor seperti usia, jenis kelamin, tinggi badan, berat badan, serta parameter fisiologis seperti detak jantung dan durasi latihan. Pendekatan konvensional sering kali tidak mempertimbangkan variasi individu secara detail, sehingga hasilnya kurang akurat.

Dataset yang digunakan adalah Calories Burnt Prediction Dataset dari Kaggle yang memuat atribut seperti jenis kelamin, usia, tinggi badan, berat badan, durasi latihan, detak jantung, dan kalori yang terbakar. Model yang digunakan dalam proyek ini antara lain Linear Regression, Random Forest Regressor, K-Nearest Neighbors (KNN), dan Algoritma Boosting. Proyek ini akan mengevaluasi performa masing-masing model berdasarkan metrik _Mean Squared Error_ (MSE), _Mean Absolute Error_ (MAE), dan RMSE. Proyek ini diharapkan tidak hanya menghasilkan model prediktif yang efektif, namun juga dapat memberikan wawasan bagi praktisi kesehatan dalam pengambilan keputusan yang berbasis data.

Model ini diharapkan dapat diaplikasikan dalam berbagai sistem monitoring kesehatan dan kebugaran seperti fitness tracker, smartwatch, atau personal health app, guna membantu pengguna mencapai target kebugaran mereka dengan cara yang efisien dan berbasis data.

## Business Understanding

### Problem Statements

- Bagaimana proses pembersihan data seperti penanganan missing value, outlier, dan duplikat dapat meningkatkan kualitas dataset serta membantu menghasilkan model pembakaran kalori yang lebih akurat dan andal?
- Bagaimana memanfaatkan data fisiologis dan aktivitas untuk memprediksi jumlah kalori yang terbakar secara akurat?
- Algoritma machine learning mana yang memberikan performa terbaik dalam memodelkan hubungan antara fitur latihan dan kalori yang terbakar?

### Goals

- Melakukan proses pembersihan data melalui identifikasi dan penanganan missing values, outlier, serta data duplikat guna meningkatkan kualitas data input, sehingga model machine learning dapat belajar dari informasi yang bersih dan representatif untuk memprediksi pembakaran kalori.
- Mengembangkan model prediktif untuk estimasi jumlah kalori yang terbakar berdasarkan atribut input.
- Membandingkan performa beberapa algoritma machine learning seperti K-Nearest Neighbors, Random Forest, Linear Regression, dan Algoritma Boosting menggunakan metrik _Mean Squared Error_ (MSE), _Mean Absolute Error_(MAE), dan RMSE.

### Solution Statement

- Menerapkan beberapa algoritma machine learning seperti K-Nearest Neighbors (KNN), Random Forest, Linear Regression, dan Algoritma Boosting untuk membangun model prediksi pembakaran kalori, dengan melakukan pembandingan performa menggunakan metrik evaluasi _Mean Squared Error_ (MSE), _Mean Absolute Error_(MAE), dan RMSE pada data latih dan uji.
- Melakukan pembersihan data secara menyeluruh sebelum pelatihan model, mencakup penanganan missing values, penghapusan duplikat, dan deteksi outlier menggunakan teknik seperti IQR (Interquartile Range).

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [_Calories Burnt Prediction_](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction) yang berasal dari kaggle yang dirancang untuk keperluan prediksi risiko diabetes berdasarkan beberapa faktor. Dataset ini terdiri dari 15000 baris dan 9 kolom, dimana setiap baris merepresentasikan satu individu. Data ini memiliki satu variabel target, yaitu "Calories" yang menunjukkan pembakaran kalori seseorang selama beraktivitas.

### Variabel-variabel pada _Calories Burnt Prediction_ adalah sebagai berikut:
- User_ID : 	Pengidentifikasi unik untuk setiap baris data atau individu. Kolom ini bersifat administratif dan tidak memiliki peran langsung dalam proses prediksi. Biasanya akan dihapus pada tahap preprocessing.
- _Gender_ : Menunjukkan jenis kelamin individu (_male_ atau _female_) yang melakukan latihan. Perbedaan gender dapat memengaruhi laju metabolisme dan jumlah kalori yang terbakar karena perbedaan fisiologis seperti massa otot dan hormon.
- _Age_ : Umur individu dalam tahun. Umur mempengaruhi tingkat metabolisme basal (BMR), yang berkontribusi pada jumlah kalori yang terbakar saat beraktivitas. Semakin tua usia seseorang, umumnya metabolisme tubuh melambat.
- _Height_ : Tinggi badan individu dalam satuan sentimeter. Tinggi badan digunakan untuk memperkirakan komposisi tubuh, seperti massa otot dan lemak tubuh, yang berpengaruh terhadap jumlah energi yang dibakar.
- _Weight_ : Berat badan individu dalam satuan kilogram. Berat badan mempengaruhi berapa banyak energi yang dibutuhkan untuk melakukan aktivitas fisik. Umumnya, orang dengan berat badan lebih tinggi akan membakar lebih banyak kalori untuk aktivitas yang sama.
- _Duration_ : Durasi latihan dalam satuan menit. Semakin lama waktu latihan, semakin banyak kalori yang terbakar. Ini merupakan salah satu fitur paling langsung yang berhubungan dengan variabel target.
- _Heart_Rate_ : Detak jantung individu selama latihan, biasanya dalam satuan denyut per menit (bpm). Detak jantung tinggi menunjukkan intensitas aktivitas yang lebih besar, dan dengan demikian, pembakaran kalori yang lebih tinggi.
- _Body_Temp_ : Suhu tubuh individu selama atau setelah latihan dalam satuan derajat Celcius. Peningkatan suhu tubuh merupakan indikator aktivitas fisik yang aktif dan metabolisme tubuh yang meningkat, yang berkorelasi dengan pembakaran kalori.
- _Calories_ :  Jumlah kalori yang terbakar oleh individu selama latihan.

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) adalah proses memahami struktur, pola, dan anomali dari sebuah dataset sebelum dilakukan pemodelan. Pada proyek ini dilakukan beberapa tahapan EDA sebagai berikut:
1. Pemeriksaan Awal pada Data
   - Memeriksa fitur data dengan `calories.info()`
     <br>Digunakan untuk melihat struktur dataset, termasuk jumlah baris dan kolom, tipe data, dan apakah ada nilai yang hilang. Hasilnya menunjukkan bahwa semua kolom memiliki tipe data numerik dan string dan tidak terdapat missing values (kosong) dengan total kolom 9 dan baris 15000.
   - Mengecek missing value pada dataset `calories.isnull().sum()`
     <br>Mengecek apakah pada dataset terdapat missing value. Hasilnya tidak ditemukan adanya missing value.
   - Mengecek duplikasi data `calories.duplicated().sum()`
     <br>Mengecek apakah ada data yang duplikat atau tidak. Untuk hasil pada pengecekan duplikasi, data tidak memiliki nilai duplikat.
   - Memastikan tidak ada missing value `calories.isna().sum()`
     <br>Memastikan bahwa tidak ada nilai _NaN_ untuk setiap kolom yang menandakan missing value.
   - Penghapusan kolom **User_ID**
     <br>Kolom **User_ID** dihapus karena tidak relevan terhadap proses prediksi, hanya berfungsi sebagai identifikasi unik.
   - Encoding fitur kategorikal kolom **Gender**
     <br>Kolom **Gender** diubah menjadi nilai numerik (Male = 1, Female = 0) dengan LabelEncoder.
2. Statistika Deskriptif
3. Deteksi Outlier dengan Boxplot
5. Penghapusan Outlier dengan IQR Method
6. Distribusi Fitur (Histogram)
7. Pairplot (Visualisasi Hubungan antar Variabel)
8. Heatmap Korelasi

## Data Preparation
Tahap Data Preparation merupakan langkah penting sebelum melakukan proses training model machine learning. Tujuannya adalah untuk memastikan data dalam kondisi optimal agar model dapat belajar secara efektif. Berikut ini adalah tahapan data preparation yang dilakukan dalam proyek ini:
1. Mengecek Ringkasan Informasi Dataset
   - Mengecek informasi data menggunakan `calories.info()`.
   - Tujuan dari pengecekan ini adalah untuk membantu memahami struktur data, Mengidentifikasi nilai yang hilang, memeriksa setiap type data dari setiap kolom, dan juga termasuk langkah awal untuk proses _data cleaning_.

2. Mengecek Duplikasi Data
   - Mengecek duplikasi data dilakukan dengan kode `calories.duplicated().sum()`.
   - Pada proyek ini tidak ditemukan adanya data yang duplikat.
   - Mengecek duplikasi data bertujuan agar data tidak ganda, data ganda dapat mendominasi hasil perhitungan statistik yang menghasilkan kesimpulan yang bias. Proses pengecekan duplikasi diperlukan untuk mendapatkan representasi data yang akurat, efisien, dan relevan untuk pengambilan keputusan yang tepat.

3. Pemeriksaan dan Penanganan Nilai Kosong (Missing Values)
   - Mengecek missing value dapat menggunakan `calories.isnull().sum()`
   - Pada proyek ini tidak ditemukan adanya missing value.
   - Nilai kosong dapat mengganggu proses pelatihan model. Jika ada, harus dilakukan penanganan seperti imputasi (pengisian nilai) atau penghapusan baris/kolom.

4. Pemeriksaan dan Penanganan Nilai kosong atau NaN
   - Mengecek missing value dapat menggunakan `calories.isna().sum()`
   - Pada proyek ini tidak ditemukan adanya NaN.
   - Nilai kosong dapat mengganggu proses pelatihan model. Jika terdapat NaN dapat dilakukan penanganan seperti imputasi (pengisian nilai) menggunakan mean atau median atau penghapusan baris/kolom.
     
5. Penghapusan Kolom **User_ID**
   - Penghapusan kolom **User_ID** dilakukan melalui kode berikut `calories.drop('User_ID', axis=1, inplace=True)`.
   - Kolom **User_ID** hanya berisi identitas dan tidak memiliki kontribusi terhadap prediksi pembakaran kalori. Kolom seperti ini disebut irrelevant feature dan dapat menyebabkan noise dalam proses pelatihan model.

6. Encoding Fitur Kategorikal kolom **Gender**
   - Encoding Fitur Kategorikal kolom **Gender** dilakukan melalui kode berikut `calories['Gender'] = calories['Gender'].map({'male': 0, 'female': 1})`.
   - Kolom **Gender** berisi jenis kelamin dan akan mudah di modeling jika kolom tersebut diubah menjadi tipe data numerik.

7. Pengecekan Outlier dengan Box-plot
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

8. Menangani Outlier (IQR)
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

9. Distribusi Fitur (Histogram)
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

10. Korelasi Antar Fitur
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

11. Splitting Data
    - Kode yang diterapkan dalam splitting data adalah sebagai berikut:
      ```python
      X = data.drop(["Outcome"],axis =1)
      y = data["Outcome"]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
      ```
    - Data terbagi menjadi 20% untuk `X_test` dan `y_test` serta 80% untuk `X_train` dan `y_train`. Melakukan stratified splitting yaitu dengan `stratify=y`.
    - Data train digunakan untuk melatih model, data test digunakan untuk menguji generalisasi model terhadap data baru.

12. Standarisasi Data training dan testing
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

