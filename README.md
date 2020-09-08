# GridsearchCV-Crossvalidation
Yapay sinir ağları oluşturarak eğittiğim model için GridsearchCV ve Cross Validation kullanarak modelin en iyi skorunu ve en iyi parametrelerini test ettim.

kc_house.csv data setini kullanarak waterfront(deniz kenarı ev) sınıflandırması yaptım.Sınıflandırma yaparken sinir ağı oluşturdum.Sinir ağında model için kullanılacak en iyi parametreleri gridsearchcv kullanarak gördüm.
GridsearchCV ile kullanmasını istediğim parametreler;
1-optimizer
2-epochs
3-batch_size
ve en iyi skoru hesaplamasını istedim.

Cross Validation ile de modelin ortalamasını ve standart sapmasını hesapladım.

Kullandığım Kütüphaneler;

Keras ve Scikit-learn

