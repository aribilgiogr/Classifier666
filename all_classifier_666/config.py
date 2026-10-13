# ---------------------------------------------------------------------------
# IMPORT BÖLÜMÜ: Kullanacağımız sınıflandırma algoritmalarını içe aktarıyoruz.
# "import" = başka bir kütüphanedeki hazır kodu bu dosyada kullanılabilir yapmak.
# ---------------------------------------------------------------------------

# LogisticRegression: Adına rağmen bir SINIFLANDIRMA algoritmasıdır; olasılık hesaplayarak sınıf tahmin eder.
# SGDClassifier: "Stochastic Gradient Descent" yöntemiyle öğrenen, hızlı ve doğrusal bir sınıflandırıcıdır.
from sklearn.linear_model import LogisticRegression, SGDClassifier

# DecisionTreeClassifier: Veriyi "evet/hayır" sorularıyla dallara ayıran karar ağacı.
from sklearn.tree import DecisionTreeClassifier

# RandomForestClassifier: Çok sayıda karar ağacını birlikte çalıştırıp oy çokluğuyla karar verir (topluluk/ensemble yöntemi).
# GradientBoostingClassifier: Ağaçları sırayla ekler; her yeni ağaç öncekinin hatalarını düzeltmeye çalışır.
# AdaBoostClassifier: Zayıf modellerin hatalı tahmin ettiği örneklere daha çok ağırlık vererek öğrenir.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# SVC: Destek Vektör Makinesi; sınıfları ayıran en geniş marjinli sınırı bulmaya çalışır.
from sklearn.svm import SVC

# KNeighborsClassifier: Yeni bir örneğin en yakın K komşusuna bakıp çoğunluğun sınıfını seçer.
from sklearn.neighbors import KNeighborsClassifier

# GaussianNB: Naive Bayes; özelliklerin normal dağıldığını varsayıp Bayes teoremiyle olasılık hesaplar.
from sklearn.naive_bayes import GaussianNB

# LDA ve QDA: Sınıfları istatistiksel dağılımlarına göre ayırır. LDA doğrusal, QDA ise eğrisel (ikinci dereceden) sınır çizer.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# MLPClassifier: Çok katmanlı yapay sinir ağı (Multi-Layer Perceptron).
from sklearn.neural_network import MLPClassifier

# LGBMClassifier: Microsoft'un geliştirdiği, çok hızlı bir gradient boosting kütüphanesi (scikit-learn dışı).
from lightgbm import LGBMClassifier

# XGBClassifier: Yarışmalarda çok popüler, güçlü bir gradient boosting kütüphanesi (scikit-learn dışı).
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# MODEL LİSTESİ: Karşılaştırmak istediğimiz tüm modelleri bir listede topluyoruz.
# Her eleman bir "demet" (tuple) yani (görünen_isim, model_nesnesi) çiftidir.
# Modellerin parantezleri BOŞ olduğu için hepsi varsayılan (default) ayarlarla çalışır.
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    # Liste başlıyor; her satır bir modeli temsil ediyor.
    ('Logistic Regression', LogisticRegression()),                      # Basit ve hızlı doğrusal sınıflandırıcı
    ('Decision Tree', DecisionTreeClassifier()),                        # Tek bir karar ağacı
    ('Random Forest', RandomForestClassifier()),                        # Birçok karar ağacından oluşan orman
    ('SVM', SVC()),                                                     # Destek vektör makinesi
    ('KNN', KNeighborsClassifier()),                                    # En yakın komşular yöntemi
    ('Naive Bayes', GaussianNB()),                                      # Olasılık tabanlı basit model
    ('Gradient Boosting', GradientBoostingClassifier()),                # Hataları adım adım düzelten ağaç topluluğu
    ('AdaBoost', AdaBoostClassifier()),                                 # Zor örneklere odaklanan boosting yöntemi
    ('SGD', SGDClassifier()),                                           # Stokastik gradyan inişiyle öğrenen doğrusal model
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),     # Doğrusal ayrım analizi
    ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),  # Eğrisel ayrım analizi
    ('MLP Neural Network', MLPClassifier()),                            # Yapay sinir ağı
    ('LightGBM', LGBMClassifier(verbose=-1)),                           # verbose=-1 -> eğitim sırasında konsola uyarı/mesaj yazdırma
    ('XGBoost', XGBClassifier())                                        # Güçlü gradient boosting modeli (son eleman, sonunda virgül gerekmez)
]  # Liste burada kapanıyor

# ---------------------------------------------------------------------------
# PARAMETRE SÖZLÜĞÜ: Hiperparametre ayarı (GridSearchCV gibi) için denenecek değerler.
# Hiperparametre = modeli eğitmeden ÖNCE bizim seçtiğimiz ayarlar (öğrenilen değil).
# NOT: tester.py bu sözlüğü şu an KULLANMIYOR; ileride ayar yapmak için hazırlanmış.
# Sözlük yapısı: { 'model_adı': { 'parametre_adı': [denenecek_değerler] } }
# ---------------------------------------------------------------------------
DEFAULT_PARAMETERS = {
    # C: Düzenlileştirme gücünün tersi. Küçük C = daha güçlü kısıtlama (aşırı öğrenmeyi azaltır).
    'Logistic Regression': {'C': [0.1, 1.0, 10.0]},

    # max_depth: Ağacın en fazla kaç seviye derinleşebileceği (None = sınırsız).
    # min_samples_split: Bir düğümün bölünebilmesi için gereken en az örnek sayısı.
    'Decision Tree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},

    # n_estimators: Ormandaki ağaç sayısı. max_depth: Her ağacın maksimum derinliği.
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 10]},

    # kernel: Verinin nasıl dönüştürüleceği; 'linear' doğrusal, 'rbf' eğrisel sınırlar üretir.
    'SVM': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},

    # n_neighbors: Kaç komşuya bakılacağı. weights: 'uniform' = hepsi eşit, 'distance' = yakın olanın oyu daha değerli.
    'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},

    # --- Aşağıdakiler sonradan eklenen modeller ---

    # var_smoothing: Varyansa eklenen küçük bir sabit; sayısal kararsızlığı önler ve modeli daha esnek/katı yapar.
    # Değerler çok küçük sayılardır (1e-9 = 0.000000001); üstel ölçekte denenir.
    'Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7]},

    # n_estimators: Eklenecek ağaç sayısı. learning_rate: Her ağacın katkısının büyüklüğü
    # (küçük değer = yavaş ama daha dikkatli öğrenme). max_depth: Her ağacın derinliği (boosting'de genelde sığ tutulur).
    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1, 0.5], 'max_depth': [2, 3, 5]},

    # n_estimators: Zayıf model sayısı. learning_rate: Her zayıf modelin ağırlığını küçültür/büyütür.
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},

    # loss: Hata fonksiyonu; 'hinge' SVM gibi, 'log_loss' lojistik regresyon gibi davranır.
    # alpha: Düzenlileştirme gücü (büyük alpha = daha basit model).
    'SGD': {'loss': ['hinge', 'log_loss'], 'alpha': [0.0001, 0.001, 0.01]},

    # solver: Hesaplama yöntemi. 'svd' varsayılandır, 'lsqr' ise en küçük kareler yöntemidir.
    # (shrinkage parametresi sadece 'lsqr'/'eigen' ile çalıştığı için, hata çıkmasın diye bilerek eklemedik.)
    'Linear Discriminant Analysis': {'solver': ['svd', 'lsqr']},

    # reg_param: Kovaryans matrisini düzenlileştirir. 0 = hiç düzenlileştirme, büyük değer = daha kararlı ama daha basit model.
    'Quadratic Discriminant Analysis': {'reg_param': [0.0, 0.1, 0.5]},

    # hidden_layer_sizes: Gizli katman yapısı. (50,) = 50 nöronlu tek katman, (50, 50) = 50'şer nöronlu iki katman.
    # activation: Nöronların aktivasyon fonksiyonu. alpha: Düzenlileştirme gücü (aşırı öğrenmeyi azaltır).
    'MLP Neural Network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]},

    # num_leaves: Ağaçtaki en fazla yaprak sayısı (LightGBM'in en önemli ayarı). Büyük değer = daha karmaşık model.
    'LightGBM': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'num_leaves': [15, 31]},

    # XGBoost'ta max_depth ağaç derinliğini, learning_rate öğrenme hızını, n_estimators ağaç sayısını belirler.
    'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1, 0.3], 'max_depth': [3, 5, 7]},
}  # Sözlük burada kapanıyor; artık DEFAULT_MODELS'taki 14 modelin hepsi için parametre var
