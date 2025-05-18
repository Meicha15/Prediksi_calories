# Laporan Proyek Machine Learning - Meicha Salsabila Budiyanti

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
     for column in calories.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=calories[column])
        plt.title(f'Boxplot for {column}')
        plt.show()
     ```
   - Hasilnya setiap kolom memiliki nilai outlier kecuali kolom Gender, Age, dan Duration.
   -  Beberapa kolom/fitur memiliki nilai outlier yang jika tidak ditangani, outlier bisa menyebabkan model belajar pola yang tidak benar (overfitting atau bias).
   -  > Gambar dapat dilihat di : [Google Collabs - project](https://colab.research.google.com/drive/1YLtZ1iAlsAYZtMj-vUt70D9CUUDr6GcU?usp=sharing)

8. Menangani Outlier (IQR)
   - Penanganan dilakukan dengan metode Interquartile Range (IQR) yang dilakukan memalui kode berikut:
     ```python
     # Ambil hanya kolom numerikal
        numeric_cols = calories.select_dtypes(include='number').columns
     # Hitung Q1, Q3, dan IQR hanya untuk kolom numerikal
        Q1 = calories[numeric_cols].quantile(0.25)
        Q3 = calories[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
     # Buat filter untuk menghapus baris yang mengandung outlier di kolom numerikal
        filter_outliers = ~((calories[numeric_cols] < (Q1 - 1.5 * IQR)) |
                    (calories[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
     # Terapkan filter ke dataset asli (termasuk kolom non-numerikal)
        calories = calories[filter_outliers]
     # Cek ukuran dataset setelah outlier dihapus
        calories.shape

     calories.info()
     ```
   - Setelah penghapusan outlier jumlah baris yang semula 15000 menjadi 14611 baris.
   - Alasan dilakukan penerapan metode IQR adalah karena ingin menghapus outlier agar nantinya tidak berpengaruh ke model.

9. Distribusi Fitur (Histogram)
   - Disitribusi fitur dilakukan melalui pembuatan histogram dengan kode berikut :
     ```python
     calories.hist(bins= 50, figsize=(15,10))
     plt.show()
     ```
   - Hasil dari distribusi nya adalah sebagai berikut:
      * Gender: Data biner (0 = perempuan, 1 = laki-laki), dengan sedikit lebih banyak laki-laki.
      * Age: Miring ke kanan; mayoritas usia 18–30 tahun.
      * Height: Hampir normal, puncak di 165–185 cm.
      * Weight: Sedikit miring ke kanan, dominan di 60–80 kg.
      * Duration: Hampir merata; variasi durasi latihan terdistribusi baik.
      * Heart_Rate: Mendekati normal, puncak di 95–100 bpm.
      * Body_Temp: Miring ke kiri; suhu banyak di kisaran 40–41°C.
      * Calories: Miring ke kanan; sebagian besar di bawah 100 kkal, ekor hingga 300 kkal.
    - Alasan dilakukannya ini adalah untuk memahami karakteristik data, mendeteksi ketidakseimbangan/skewness, serta menentukan perlunya transformasi atau penyesuaian model.

10. Korelasi Antar Fitur
   - Menggunakan correlation matrix dan pairplot
     ```python
     #Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
     sns.pairplot(calories, diag_kind = 'kde')
     #Untuk mengevaluasi skor korelasinya, gunakan fungsi corr()
     plt.figure(figsize=(10, 8))
     correlation_matrix = calories.corr().round(2)
     # Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
     sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
     plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
     ```
   - Hasil dari correlation matrix adalah
      * Duration: Korelasi 0.95, arah korelasi positif sangat kuat. Artinya semakin lama durasi latihan, semakin banyak kalori terbakar.
      * Heart Rate: Korelasi 0.90, arah korelasi positif sangat kuat. Artinya detak jantung lebih tinggi menunjukkan intensitas latihan yang lebih tinggi dan pembakaran kalori lebih besar.
      * Body Temperature: Korelasi 0.84, arah korelasi positif kuat. Artinya suhu tubuh meningkat saat aktivitas berat, yang sejalan dengan kalori yang terbakar.
      * Age: Korelasi 0.16, arah korelasi positif lemah. Artinya bertambahnya usia sedikit berkaitan dengan peningkatan kalori terbakar, kemungkinan karena variasi metabolisme.
      * Weight: Korelasi 0.04, arah korelasi positif sangat lemah. Artinya sedikit korelasi positif, mengindikasikan berat badan bukan faktor utama dalam pembakaran kalori dalam dataset ini.
      * Height: Korelasi 0.02, arah korelasi positif sangat lemah. Artinya tinggi badan memiliki pengaruh yang sangat kecil terhadap kalori terbakar.
      * Gender: Korelasi -0.02, arah korelasi negatif sangat lemah. Artinya jenis kelamin hampir tidak berpengaruh langsung terhadap kalori yang terbakar.
   - Analisis korelasi digunakan untuk mengidentifikasi fitur-fitur yang paling relevan dalam membangun model prediksi kalori terbakar. Dalam kasus ini, fitur dengan korelasi tinggi terhadap variabel target seperti Duration, Heart Rate, dan Body Temperature dianggap sangat penting karena memiliki kontribusi besar terhadap performa model. Di sisi lain, korelasi tinggi antara beberapa fitur seperti Height dan Weight (dengan nilai korelasi sebesar 0.96) menunjukkan adanya multikolinearitas, yaitu kondisi di mana dua fitur sangat saling berkaitan, yang dapat memengaruhi kestabilan model linear seperti regresi linier, namun tidak terlalu berdampak pada model non-linier seperti Random Forest. Sementara itu, fitur dengan korelasi rendah seperti Gender dan Height tetap dipertahankan dalam model karena meskipun kontribusinya kecil secara individu, keberadaannya tetap dapat meningkatkan akurasi prediksi ketika digunakan bersama fitur-fitur lain melalui interaksi kompleks antar variabel.

11. Splitting Data
    - Kode yang diterapkan dalam splitting data adalah sebagai berikut:
      ```python
      #Splitting dataset 80:20
      X = calories.drop('Calories', axis=1)
      y = calories['Calories']
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

      print(f'Total # of sample in whole dataset: {len(X)}')
      print(f'Total # of sample in train dataset: {len(X_train)}')
      print(f'Total # of sample in test dataset: {len(X_test)}')
      ```
    - Data terbagi menjadi 20% untuk `X_test` dan `y_test` serta 80% untuk `X_train` dan `y_train`.
    - Data train digunakan untuk melatih model, data test digunakan untuk menguji generalisasi model terhadap data baru.
    - Total # of sample in whole dataset: 14611 (Total keselurahan data pada dataset)
    - Total # of sample in train dataset: 11688 (Total data uji)
    - Total # of sample in test dataset: 2923 (Total data test)

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
1. K-Nearest Neighbors (KNN)
   Parameter yang digunakan: n_neighbors = 2. Cara Kerja: KNN melakukan prediksi berdasarkan rata-rata nilai target dari sejumlah k titik data terdekat dalam ruang fitur. Jarak antar titik biasanya dihitung menggunakan jarak Euclidean.
   * Kelebihan:
      - Non-parametrik (tidak mengasumsikan bentuk distribusi data).
      - Sederhana dan mudah dipahami.
      - Efektif dalam menangkap pola lokal.
   * Kekurangan:
      - Sangat sensitif terhadap skala data dan outlier.
      - Kurang efisien untuk dataset besar karena perlu menghitung jarak terhadap seluruh data pelatihan.
      - Tidak cocok untuk data dengan dimensi tinggi tanpa reduksi fitur.
     ```python
     knn = KNeighborsRegressor(n_neighbors=2)
     knn.fit(X_train, y_train)
     #mse knn
     models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
     #mae knn
     models.loc['train_mae','knn'] = mean_absolute_error(y_pred = knn.predict(X_train), y_true=y_train)
     #rmse knn
     models.loc['train_rmse','knn'] = np.sqrt(mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train))
     ```

2. Random Forest
   - Parameter yang digunakan :
     * `n_estimators = 50` : jumlah pohon dalam hutan.
     * `max_depth = 16` : kedalaman maksimum pohon.
     * 'n_jobs=-1' gunakan semua core yang tersedia pada CPU.
     * `random_state = 55` : untuk menjaga hasil tetap konsisten.
   - Cara Kerja : Random Forest membentuk banyak pohon keputusan (decision tree) dan menggabungkan hasilnya (rata-rata untuk regresi) agar lebih stabil dan akurat. Random Forest juga menggunakan subset fitur dan data (bagging) untuk membangun tiap pohon.
   - Kelebihan : Mampu menangkap hubungan non-linier, tidak sensitif terhadap outlier dan multikolinearitas, bias rendah dan akurasi tinggi.
   - Kekurangan : Waktu komputasi bisa tinggi, apalagi jika pohon sangat dalam, lebih sulit diinterpretasi dibanding regresi linear.
     ```python
     RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
     RF.fit(X_train, y_train)
     #mse rf
     models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
     #mae rf
     models.loc['train_mae','RandomForest'] = mean_absolute_error(y_pred=RF.predict(X_train), y_true=y_train)
     #rmse rf
     models.loc['train_rmse','RandomForest'] = np.sqrt(mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train))
     ```
3. Algoritma Boosting
   - Parameter yang digunakan :
     * 'learning_rate=0.05' : untuk mengontrol seberapa besar kontribusi tiap model kecil (weak learner) dalam proses boosting.
     * `random_state = 55` : untuk menjaga hasil tetap konsisten.
   - Cara Kerja : Bekerja dengan melatih model secara berurutan, di mana setiap model baru berfokus untuk memperbaiki kesalahan prediksi dari model sebelumnya dengan memberikan bobot lebih besar pada data yang sulit diprediksi, sehingga menghasilkan prediksi akhir yang lebih akurat melalui kombinasi berbobot dari semua model.
   - Kelebihan : Mampu meningkatkan akurasi model sederhana menjadi model yang kuat, serta mampu dalam menangani hubungan non-linier secara efektif.
   - Kekurangan : ensitif terhadap data outlier dan noise karena terus memberi bobot lebih besar pada kesalahan, serta membutuhkan waktu pelatihan yang lebih lama dibanding metode paralel seperti Random Forest karena proses pembelajarannya dilakukan secara berurutan.
     ```python
     boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
     boosting.fit(X_train, y_train)
     #mse bosting
     models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
     #mae boosting
     models.loc['train_mae','Boosting'] = mean_absolute_error(y_pred=boosting.predict(X_train), y_true=y_train)
     #rmse boosting
     models.loc['train_rmse','Boosting'] = np.sqrt(mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train))
     ```
5. Linear Regression
   - Parameter yang digunakan : `sklearn.linear_model.LinearRegression()`.
   - Cara Kerja : Linear regression mencari garis lurus terbaik yang meminimalkan jumlah kuadrat kesalahan antara prediksi dan nilai sebenarnya. Model ini mengasumsikan hubungan linier antara fitur dan target.
   - Kelebihan : Mudah diinterpretasikan, membutuhkan waktu komputasi yang cepat.
   - Kekurangan : Tidak mampu menangkap hubungan non-linear, sensitif terhadap multikolinearitas, Asumsi normalitas dan homoskedastisitas sering tidak terpenuhi dalam data nyata.
     ```python
     lr = LinearRegression()
     lr.fit(X_train, y_train)
     #mse linear regression
     models.loc['train_mse','LinearRegression'] = mean_squared_error(y_pred=lr.predict(X_train), y_true=y_train)
     #mae linear regression
     models.loc['train_mae','LinearRegression'] = mean_absolute_error(y_pred=lr.predict(X_train), y_true=y_train)
     #rmse linear regression
     models.loc['train_rmse','LinearRegression'] = np.sqrt(mean_squared_error(y_pred=lr.predict(X_train), y_true=y_train))
     ```
Model terbaik yang dipilih adalah Random Forest, karena menghasilkan nilai _Mean Squared Error_ (MSE), _Mean Absolute Error_ (MAE), dan RMSE terkecil pada data uji, yaitu sebesar 0.001263, 0.000716, dan 0.001124 dibandingkan dengan model KNN, Linear Regression, dan Algoritma Boosting. Selain itu, model ini juga menunjukkan performa yang konsisten antara data latih dan data uji, menandakan kemampuan generalisasi yang baik tanpa overfitting. Random Forest juga unggul dalam menangkap hubungan non-linier antar fitur, sehingga lebih efektif dalam menyelesaikan permasalahan regresi pada dataset ini.Hasil algoritma yang terbaik berdasarkan metrik yang diperoleh.

## Evaluation

Dalam proyek ini digunakan tiga metrik evaluasi utama untuk masalah regresi, yaitu **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, dan **Root Mean Squared Error (RMSE)**. Ketiganya mengukur sejauh mana nilai prediksi dari model berbeda dari nilai aktual.

Berikut adalah rumus dari masing-masing metrik:

- **Mean Squared Error (MSE)**
    
  $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Mean Absolute Error (MAE)**
    
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

- **Root Mean Squared Error (RMSE)**
    
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Keterangan:** 
- $y_i$ = nilai aktual  
- $\hat{y}_i$ = nilai prediksi  
- ${n}$ = jumlah data  

---

# Hasil Evaluasi Model

Tabel berikut menyajikan hasil evaluasi untuk empat model regresi: **KNN**, **Random Forest (RF)**, **Boosting**, dan **Linear Regression** berdasarkan nilai MSE, MAE, dan RMSE pada data train dan test:

| Model               | Train MSE | Test MSE | Train MAE | Test MAE | Train RMSE | Test RMSE |
|---------------------|-----------|----------|-----------|----------|-------------|------------|
| **KNN**             | 0.014856  | 0.039048 | 0.002888  | 0.004680 | 0.003854    | 0.006249   |
| **Random Forest**   | 0.001263  | 0.008322 | 0.000716 | 0.001824 | 0.001124    | 0.002885   |
| **Boosting**        | 0.197413  | 0.197314 | 0.010367  | 0.010491 | 0.01405    | 0.014047   |
| **Linear Regression** | 0.125898 | 0.126514 | 0.008303  | 0.008526 | 0.011224    | 0.011248   |

---

# Kesimpulan

Berdasarkan hasil evaluasi:

- **Model Random Forest (RF)** menunjukkan performa terbaik. Hal ini terlihat dari nilai MSE, MAE, dan RMSE yang paling rendah baik pada data training maupun testing, menunjukkan kemampuan model dalam melakukan generalisasi terhadap data baru.
- **Model KNN** juga memberikan performa yang cukup baik, meskipun sedikit lebih buruk dibanding RF, terutama pada data test.
- **Model Boosting** dan **Linear Regression** memiliki performa yang relatif rendah. Kedua model menunjukkan error yang cukup besar pada semua metrik evaluasi.
- Perbedaan besar antara nilai error Linear Regression/Boosting dan Random Forest/KNN menunjukkan bahwa model berbasis ensemble seperti RF jauh lebih mampu menangani kompleksitas data.

---




## Daftar Referensi
> [1]World Health Organization, Physical activity, June 26, 2024. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/physical-activity

