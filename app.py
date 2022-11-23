import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel 
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5 import QtGui
import os
import glob
import pandas as pd
import statistics
import numpy as np 
import scipy
import pylab 
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import ttest_ind
from testy import Proba


class Fraktal_app(QMainWindow):
    """Główna klasa aplkacji"""
    def __init__(self):
        super().__init__()
        # Import interfejsu użytkownika bezpośrednio
        # z pliku sekwencje.ui
        loadUi("fraktal.ui", self)
        self.show()
        # Powiązanie sygnałów (akcji wywoływanych na widżetach) 
        # i slotów (funkcji)
        self.actionDane1.triggered.connect(self.wybierz_plik)
        self.actionDane2.triggered.connect(self.wybierz_drugi_plik)
        self.Zakoncz.clicked.connect(self.zakoncz)
        self.QQ.clicked.connect(self.qq)
        self.Histogram.clicked.connect(self.histogram)     
        self.SWtest.clicked.connect(self.SW_test)
        self.Ftest.clicked.connect(self.F_test)
        self.Ttest.clicked.connect(self.ttest)
        self.dane_zaladowane = False
    

    def wybierz_plik(self):
        """Pobiera dane z wybranego folderu i tworzy z nich scaloną tabelę"""
        #wybieranie folderu, w którym mieszczą się pomiary jednej grupy badawczej
        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2')
        
        self.input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', sciezka)
        sciezka = os.path.join(katalog, 'fraktal_2', self.input_dir)
        os.chdir(sciezka)
        plik = self.input_dir + ".csv"
        

        #sprawdzenie czy połączony plik ju istnieje, jeśli nie - tworzy go
        if os.path.exists(plik):
            print("Znaleziono połączone dane.")
            self.QQ.setEnabled(True)
            self.Histogram.setEnabled(True)
            self.SWtest.setEnabled(True)
            

        else:
            extension = 'csv'
            pliki = [i for i in glob.glob('*.{}'.format(extension))]
            polaczone_plik = pd.concat([pd.read_csv(f) for f in pliki ])
            polaczone_plik.to_csv( self.input_dir + ".csv", index=False, encoding='utf-8-sig')
            #składanie wierszy z analizy fraktalnej
            dane = pd.read_csv(self.input_dir + '.csv')
            fraktal_plik = dane[dane['Area'].isin(['0'])] 
            print(fraktal_plik)
            #aktywowanie wcześniej wyłączonych przycisków
            self.QQ.setEnabled(True)
            self.Histogram.setEnabled(True)
            self.SWtest.setEnabled(True)

    def wybierz_drugi_plik(self):
        """Pobiera dane z wybranego folderu i tworzy z nich scaloną tabelę"""
        #analogiczne postępowanie z drugim wybranym folderem, z którego pochodzą dane z drugiej próby badawczej
        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2')
        self.input_dir2 = QFileDialog.getExistingDirectory(None, 'Select a folder:', sciezka)
        sciezka = os.path.join(katalog, 'fraktal_2', self.input_dir2)
        os.chdir(sciezka)
        plik = self.input_dir2 + ".csv"

        if os.path.exists(plik):
            print("Znaleziono połączone dane.")
            self.QQ.setEnabled(True)
            self.Histogram.setEnabled(True)
            self.SWtest.setEnabled(True)
            self.Ftest.setEnabled(True)
            self.Ttest.setEnabled(True)
        else:
            extension = 'csv'
            pliki = [i for i in glob.glob('*.{}'.format(extension))]
            polaczone_plik = pd.concat([pd.read_csv(f) for f in pliki ])
            polaczone_plik.to_csv( self.input_dir2 + ".csv", index=False, encoding='utf-8-sig')
            #składanie wierszy z analizy fraktalnej
            dane = pd.read_csv(self.input_dir2 + '.csv')
            fraktal_plik = dane[dane['Area'].isin(['0'])] 
            print(fraktal_plik)
            self.QQ.setEnabled(True)
            self.Histogram.setEnabled(True)
            self.Ftest.setEnabled(True)
            self.Ttest.setEnabled(True)
            
    def histogram(self):
        """Histogram będzie ilustrował rozkład wartości analizy fraktalnej dla pojedynczych komórek. 
        Jest on potrzebny do sprawdzenia normalności rozkładu i przeprowadzenia testu T-Studenta"""
        #budowanie obiektu z klasy z testami - histogram 
        plik = self.input_dir
        plik1 = Proba(plik)
        plik1.histogram()
        #zapis wykresu


        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2')
        os.chdir(sciezka)

        plt.savefig(sciezka + '/wykresy/wykres_histogram_1.jpeg', bbox_inches='tight')
        plt.figure()

        #reset przy zmianie danych
        plt.figure()

        #otwieranie obrazu wykresu w aplikacji
        try:
            wykres =  sciezka + "/wykresy/wykres_histogram_1.jpeg"

            self.Wykres1.setPixmap(QtGui.QPixmap(wykres))
            self.Wykres1.setScaledContents(True)
            self.Wykres1.setObjectName("Wykres3")
            self.verticalLayout.addWidget(self.Wykres1)

            tekst = '-  Histogram będzie ilustrował rozkład wartości analizy fraktalnej dla pojedynczych komórek.\n\
    Jest on potrzebny do sprawdzenia normalności rozkładu i przeprowadzenia testu T-Studenta'
            self.info.setText(tekst)
            plt.figure()


        except:
            print('Błąd. wykres_histogram_1.jpeg nie istnieje.')
     
        
       #analogiczne postępowanie dla drugiej grupy badawczej
        plik = self.input_dir2
        plik2 = Proba(plik)
        plik2.histogram()


        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2')
        os.chdir(sciezka)
        plt.savefig(sciezka + '/wykresy/wykres_histogram_2.jpeg', bbox_inches='tight')
        plt.figure()
        
        
        #otwieranie obrazu wykresu w aplikacji
        

        try:
            wykres = sciezka + "/wykresy/wykres_histogram_2.jpeg"
            self.Wykres2.setPixmap(QtGui.QPixmap(wykres))
            self.Wykres2.setScaledContents(True)
            self.Wykres2.setObjectName("Wykres4")
            self.verticalLayout.addWidget(self.Wykres2)

        except:
            print("Błąd. wykres_histogram_2.jpeg nie istnieje")



    def qq(self):
        """Wykres kwantyl-kwantyl ilustruje normalność rozkładu dla każdej grupy z osobna, 
        jest potrzebny aby przeprowadzić test T-Studenta"""
        # QQPLOT (jeżeli punkty wykresu leżą blisko prostej i są równomiernie rozłożone 
        # po jej jednej i drugiej stronie (np. naprzemiennie), to dane pochodzą z rozkładu normalnego.)
        
        #tworzenie obiektu z klasy z testami - wykres quantile-quantile 
        plik = self.input_dir
        plik1 = Proba(plik)
        plik1.qq_wykres()

        #zapis wykresu qq i wyświetlenie go w aplikacji
        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2')
        os.chdir(sciezka)

        plt.savefig(sciezka + '/wykresy/wykresqq_1.jpeg', bbox_inches='tight')
        plt.figure()
        wykres = sciezka + "/wykresy/wykresqq_1.jpeg"
        
        self.Wykres1.setPixmap(QtGui.QPixmap(wykres))
        self.Wykres1.setScaledContents(True)
        self.Wykres1.setObjectName("Wykres1")
        self.verticalLayout.addWidget(self.Wykres1)

        #analogiczne postępowanie dla drugiej grupy badawczej
                
        plik = self.input_dir2
        plik2 = Proba(plik)
        plik2.qq_wykres()
        
        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2')
        os.chdir(sciezka)



        plt.savefig(sciezka + '/wykresy/wykresqq_2.jpeg', bbox_inches='tight')
        plt.figure()
        wykres = sciezka + "/wykresy/wykresqq_2.jpeg"
        self.Wykres2.setPixmap(QtGui.QPixmap(wykres))
        self.Wykres2.setScaledContents(True)
        self.Wykres2.setObjectName("Wykres2")
        self.verticalLayout.addWidget(self.Wykres2)


        tekst = '-  QQPLOT (jeżeli punkty wykresu leżą blisko prostej i są równomiernie rozłożone \n\
    po jej jednej i drugiej stronie (np. naprzemiennie), to dane pochodzą z rozkładu normalnego.\n\
    Wykres kwantyl-kwantyl ilustruje normalność rozkładu dla każdej grupy z osobna,\n\
    jest potrzebny aby przeprowadzić test T-Studenta.)'
        self.info.setText(tekst)
        plt.figure()


    def zakoncz(self):
        """Wyłącza aplikację"""
        self.close()

    def SW_test(self):
        """Test słuzący do oceny czy zebrane wyniki dla każdej grupy badanej posiadają rozkład normalny"""
        #Test ten jest stosowany przeważnie do mniejszych grup (N < 100). W przypadku większych prób używamy #testu Kołmogorowa-Smirnowa (jako jeden z przedziałów podaje się N > 100).
        #Hipoteza zerowa tego testu mówi nam o tym, że nasza próba badawcza pochodzi z populacji o normalnym rozkładzie.
        #tworzenie obiektu z klasy z testami - test Shapiro-Wilka
        plik = self.input_dir
        plik1 = Proba(plik)
        plik1.test_shapiro_wilka()


        #wyświetlanie wyniku testu w aplikacji
        if 'przypomina':
            tekst = '  Próba 1 : Rozkład przypomina rozkład Gaussa (nie można odrzucic H0)\n\
                  UWAGA! Można przeprowadzić test T-Studenta.'

            self.Dane1.setText(tekst)
            plt.figure()


            tekst = '-  Test słuzący do oceny czy zebrane wyniki dla każdej grupy badanej posiadają rozkład normalny.'
            self.info.setText(tekst)
            plt.figure()

            
        elif 'nie_przypomina':
            tekst = '  Próba 1 : Rozkład nie przypomina rozkładu Gaussa (należy odrzucić H0)\n\
                  UWAGA! Nie można przeprowadzić testu T-Studenta.'
            self.Dane1.setText(tekst)
            plt.figure()

            tekst = '-  Test słuzący do oceny czy zebrane wyniki dla każdej grupy badanej posiadają rozkład normalny.'
            self.info.setText(tekst)
            plt.figure()
            

        #analogiczne postępowanie dla 2 grupy badawczej
        plik = self.input_dir2
        plik2 = Proba(plik)
        plik2.test_shapiro_wilka()

        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2')
        os.chdir(sciezka)

        if 'przypomina':
            tekst = '  Próba 2 : Rozkład przypomina rozkład Gaussa (nie można odrzucic H0)\n\
                  UWAGA! Można przeprowadzić test T-Studenta.'
            self.Dane2.setText(tekst)
           
        elif 'nie_przypomina':
            tekst = '  Próba 2 : Rozkład nie przypomina rozkładu Gaussa (należy odrzucić H0)\n\
                  UWAGA! Nie można przeprowadzić testu T-Studenta.'
            self.Dane2.setText(tekst)
          
  

    def F_test(self): 
        """Test Fishera-Snedecora - Test ten służy do weryfikacji hipotezy o równości wariancji badanej zmiennej w dwóch grupach."""
        def f_test(x,y): 
            #2 grupy
            x = np.array(x)
            y = np.array(y)
            #oblicz statystyki testu F.
            f = np.var(x, ddof=1)/np.var(y, ddof=1)
            #zdefiniuj licznik stopni swobody  
            dfn = x.size-1 
            #zdefiniuj mianownik stopni swobody
            dfd = y.size-1 
            #znajdź wartość p statystyki testu F
            p = 1-scipy.stats.f.cdf(f, dfn, dfd) 
            print(f'TEST F-SCEDECORA: f = {f}, p = {p}')
            #zwróć wartości
            return f,p
        
        #odczytywanie złożonych wcześniej tabeli, wyciągnięcie tylko wersów z analizy fraktalnej
        plik = self.input_dir + '.csv'
        plik2 = self.input_dir2 + '.csv'
        test1 = pd.read_csv(plik)
        test2 = pd.read_csv(plik2)
        fraktal1 = test1[test1['Area'].isin(['0'])]
        fraktal2 = test2[test2['Area'].isin(['0'])]    
        a = list(fraktal1.D) 
        b = list(fraktal2.D)
        f_test(a, b)
        print(list(f_test(a, b))[-1])

        #wynik przeprowadzonego testu i wyświetlanie go w aplikacji
        if list(f_test(a, b))[-1] > 0.5:
            
            tekst = '        Wariancje populacji są równe.\n\
        UWAGA! Można przeprowadzić test T-Studenta.'
            plt.figure()
            self.Dane1.setText(tekst)
            plt.figure()
            self.Dane2.setText('')

            text = '-   Test Fishera-Snedecora - Test ten służy do weryfikacji hipotezy o równości wariancji \n\
            badanej zmiennej w dwóch grupach.'
            self.info.setText(text)
            plt.figure()

        else:
            tekst2 = '        Wariancje nie są równe.\n\
        UWAGA! Nie można przeprowadzić testu T-Studenta.'
            plt.figure()
            self.Dane1.setText(tekst2)
            self.Dane2.setText('')
            text = '-   Test Fishera-Snedecora - Test ten służy do weryfikacji hipotezy o równości wariancji\n\
            badanej zmiennej w dwóch grupach.'
            self.info.setText(text)
            plt.figure()

    
    
    
    def ttest(self):

        """Test T-Studenta dla 2 niezależnych prób - przeprowadzany w celu porównania średnich z dwóch niezależnych od siebie grup.
        Wykorzystujemy go gdy chcemy porównać dwie grupy pod względem jakiejś zmiennej ilościowej 
        (wyniku analizy fraktalnej mówiącej o złożoności wypustek mikrogleju""" 
        #Hipoteza zerowa - nie ma różnicy między próbami
        #Hipoteza alternatywna = grupy się różnią


        #"wycinanie" potrzebnych danych z analizy fraktalnej z wcześniej utworzonej tabeli
        plik = self.input_dir + '.csv'
        plik2 = self.input_dir2 + '.csv'
        test1 = pd.read_csv(plik)
        test2 = pd.read_csv(plik2)
        fraktal1 = test1[test1['Area'].isin(['0'])]
        fraktal2 = test1[test1['Area'].isin(['0'])]

 
        #obliczanie odchylenia standardowego
        std1 = np.std(fraktal1.D, ddof=1)
        std2 = np.std(fraktal2.D, ddof=1)
        #obliczenie średniej dla obu grup
        mean1 = fraktal1.D.mean()
        mean2 = fraktal2.D.mean()
        #obliczanie błędu
        se1 = stats.sem(fraktal1.D)
        se2 = stats.sem(fraktal2.D)

        #test T-Studenta (ze wzoru)
        SED = np.sqrt(se1**2 + se2**2)
        t_stat= abs(mean1-mean2)/SED 
        df = len(fraktal1.D) + len(fraktal2.D) - 2
        #poziom istotności
        alpha = 0.05
        cv = stats.t.ppf(1.0 - alpha, df)
        p = (1 - stats.t.cdf(abs(t_stat), df)) * 2
        #sprawdzenie wyniku i wyświetlenie go w aplikacji
        if p > alpha:
            tekst = '  Test t-Studenta: Nie moża odrzucić H0. Nie ma różnicy między próbami'
            plt.figure()
            self.Dane1.setText(tekst)
            plt.figure()
            self.Dane2.setText('')
            plt.figure()

            tekst = '-   Test T-Studenta dla 2 niezależnych prób - przeprowadzany w celu porównania\n\
    średnich z dwóch niezależnych od siebie grup. Wykorzystujemy go gdy chcemy porównać\n\
    dwie grupy pod względem jakiejś zmiennej ilościowej wyniku analizy fraktalnej mówiącej \n\
    o złożoności wypustek mikrogleju'

            self.info.setText(tekst)
            plt.figure()

        else:
            tekst2 = '  Test t-Studenta: Należy odrzucić H0. Jest różnica między próbami'
            plt.figure()
            self.Dane1.setText(tekst2)
            plt.figure()
            self.Dane2.setText('')  
            plt.figure()

            tekst = '-   Test T-Studenta dla 2 niezależnych prób - przeprowadzany w celu porównania\n\
    średnich z dwóch niezależnych od siebie grup. Wykorzystujemy go gdy chcemy porównać dwie\n\
    grupy pod względem jakiejś zmiennej ilościowej wyniku analizy fraktalnej mówiącej \n\
    o złożoności wypustek mikrogleju'
            
            self.info.setText(tekst)
            plt.figure()
           

if __name__ == "__main__":
    """Główna funkcja aplikacji"""
    # Uruchomienie obiektu aplikacji
    aplikacja = QApplication(sys.argv)
    # Tworzenie okna
    okno = Fraktal_app()  
    # WYświetlenie okna
    okno.show()
    # Przy zamknięciu aplikacji, zwalnia się pamięć komputera
    sys.exit(aplikacja.exec_())