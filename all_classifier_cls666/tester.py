# ---------------------------------------------------------------------------
# IMPORT BÖLÜMÜ
# ---------------------------------------------------------------------------

# warnings: Python'un uyarı mesajlarını yönetmemizi sağlayan standart kütüphane.
import warnings

# pandas: Tablo (DataFrame) biçiminde veri işlemek için kullanılan kütüphane. "pd" kısaltmasıyla çağırılır.
import pandas as pd
from sklearn import clone

# train_test_split: Veriyi "eğitim" ve "test" olarak ikiye bölen fonksiyon.
from sklearn.model_selection import train_test_split, GridSearchCV

# StandardScaler: Sayısal özellikleri ortalaması 0, standart sapması 1 olacak şekilde ölçekler.
from sklearn.preprocessing import StandardScaler

# accuracy_score: Doğru tahmin oranını verir.
# classification_report: Precision, Recall, F1 gibi detaylı metrikleri raporlar.
from sklearn.metrics import accuracy_score, classification_report

from all_classifier_666.config import DEFAULT_PARAMETERS
# config.py dosyamızdan hazırladığımız model listesini alıyoruz.
from config import DEFAULT_MODELS

# Uyarı mesajlarını gizle: Eğitim sırasında çıkan "convergence warning" gibi mesajlar ekranı kirletmesin diye.
# (Dikkat: Gerçek projelerde uyarılar önemli bilgi taşıyabilir, bilinçli kullanın.)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ANA FONKSİYON: Tüm modelleri eğitir, test eder ve sonuçları tablo olarak döndürür.
# Parametrelerin yanındaki ": float" gibi ifadeler "tip ipucu"dur (beklenen veri türünü belirtir).
# "-> pd.DataFrame" fonksiyonun bir DataFrame döndüreceğini söyler.
# Eşittir ile verilen değerler (test_size=.3 gibi) VARSAYILAN değerlerdir; çağırırken verilmezse bunlar kullanılır.
# ---------------------------------------------------------------------------
def test_data(X, y, test_size: float = .3, random_state: int = 42, scale_data: bool = True,
              use_grid_search: bool = False, cv: int = 3) -> pd.DataFrame:
    """
    Test multiple classification models and return their performance metric values.

    :param X: Feature matrix
    :param y: Target vector
    :param test_size: Proportion of the dataset to include in the test set
    :param random_state: Random seed for reproducibility
    :param scale_data: Whether to scale the data before fitting models
    :return:
        DataFrame: Results comparing all classifiers
    """
    # Yukarıdaki üç tırnaklı metne "docstring" denir: fonksiyonun ne yaptığını anlatan resmi dokümantasyondur.
    # X = özellikler (girdiler), y = hedef (tahmin etmeye çalıştığımız sınıf etiketi).

    # Veriyi ikiye bölüyoruz:
    #   - Eğitim seti: Modelin öğrendiği kısım.
    #   - Test seti: Modelin hiç görmediği, başarısını ölçtüğümüz kısım.
    # test_size=0.3 -> verinin %30'u test için ayrılır.
    # random_state=42 -> her çalıştırmada aynı bölme elde edilsin (tekrarlanabilirlik).
    # stratify=y -> sınıf oranları eğitim ve testte aynı kalsın (dengeli dağılım).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)

    # Eğer ölçekleme isteniyorsa aşağıdaki blok çalışır.
    if scale_data:
        # Ölçekleyici nesnesini oluşturuyoruz.
        scaler = StandardScaler()

        # fit_transform: Ortalama ve standart sapmayı SADECE eğitim verisinden öğrenir ve eğitimi dönüştürür.
        X_train = scaler.fit_transform(X_train)

        # transform: Test verisini, eğitimden öğrenilen aynı değerlerle dönüştürür.
        # Neden fit yok? Test verisinden bilgi sızarsa (data leakage) sonuçlar olduğundan iyi görünür.
        X_test = scaler.transform(X_test)

    # Her modelin sonucunu biriktireceğimiz boş liste.
    results = []

    # Listedeki her (isim, model) çiftini sırayla dolaşıyoruz. Tuple açma (unpacking) kullanılıyor.
    for name, base_model in DEFAULT_MODELS:
        # try-except: Bir model hata verirse tüm program çökmesin, diğer modellere devam edilsin.
        try:
            grid_model = clone(base_model)
            best_params = None
            # Modeli eğitim verisiyle eğit (öğrenme aşaması).
            if use_grid_search and name in DEFAULT_PARAMETERS:
                grid_search = GridSearchCV(
                    estimator=grid_model,
                    param_grid=DEFAULT_PARAMETERS[name],
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                model = grid_model
                model.fit(X_train, y_train)

            # Eğitilen modelle test verisi için tahmin üret.
            y_pred = model.predict(X_test)

            # Doğruluk (accuracy): Doğru tahmin sayısı / toplam tahmin sayısı.
            acc = accuracy_score(y_test, y_pred)

            # Detaylı raporu sözlük (dict) formatında al ki içindeki değerlere kod ile erişebilelim.
            report = classification_report(y_test, y_pred, output_dict=True)

            # Bu modelin sonuçlarını bir sözlük olarak listeye ekle.
            results.append({
                'Classifier': name,  # Modelin adı
                'Accuracy': acc,  # Genel doğruluk
                # Precision (kesinlik): "Pozitif dedim" dediklerimin kaçı gerçekten doğruydu?
                # 'weighted avg': Sınıfların örnek sayısına göre ağırlıklı ortalama.
                'Precision': report['weighted avg']['precision'],
                # Recall (duyarlılık): Gerçekte pozitif olanların kaçını yakalayabildim?
                'Recall': report['weighted avg']['recall'],
                # F1-Score: Precision ve Recall'ın dengeli (harmonik) ortalaması.
                'F1-Score': report['weighted avg']['f1-score'],
                'Best Parameters': best_params
            })  # Sözlük ve append işlemi burada bitiyor

        # Herhangi bir hata olursa buraya düşer; hata nesnesi "e" değişkenine atanır.
        except Exception as e:
            # Hangi modelde ne hata olduğunu ekrana yazdır (f-string ile değişkenler metne gömülür).
            print(f"Error with {name}: {str(e)}")

            # Hata veren model için boş (None) değerlerle bir satır ekle ki tabloda görünsün.
            results.append({
                'Classifier': name,
                'Accuracy': None,
                'Precision': None,
                'Recall': None,
                'F1-Score': None,
                'Best Parameters': None
            })

    # Sözlük listesini tabloya (DataFrame) çevir: her sözlük bir satır olur.
    result_df = pd.DataFrame(results)

    # Doğruluğa göre büyükten küçüğe sırala (en iyi model en üstte).
    # reset_index(drop=True): Sıralamadan sonra bozulan satır numaralarını 0,1,2... diye yeniden düzenler.
    # Not: None değerler sıralamada en sona atılır.
    result_df = result_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    # Sonuç tablosunu fonksiyonu çağıran yere geri gönder.
    return result_df


# ---------------------------------------------------------------------------
# ÇALIŞTIRMA BLOĞU
# Bu koşul, dosya DOĞRUDAN çalıştırıldığında True olur (python tester.py).
# Başka bir dosyadan "import" edilirse bu blok çalışmaz; böylece dosya hem kütüphane hem betik olarak kullanılabilir.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Iris: Çiçek türlerini sınıflandırmak için kullanılan klasik küçük örnek veri seti.
    from sklearn.datasets import load_iris

    # Tabloyu yazdırırken tüm sütunlar görünsün (None = sınırsız).
    pd.set_option('display.max_columns', None)

    # Tabloyu yazdırırken tüm satırlar görünsün.
    pd.set_option('display.max_rows', None)

    # Veri setini yükle.
    iris = load_iris()

    # iris.data: Özellikler (çiçek yaprağı/çanak yaprağı ölçüleri).
    X = iris.data

    # iris.target: Hedef (çiçek türü: 0, 1 veya 2).
    y = iris.target

    # Tüm modelleri test et ve sonuç tablosunu al.
    df = test_data(X, y,use_grid_search=True)

    # Sonuç tablosunu ekrana yazdır.
    print(df)
