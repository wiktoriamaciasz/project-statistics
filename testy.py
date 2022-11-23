import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import glob
import pylab 
from scipy.stats import shapiro
import numpy as np
import scipy

class Proba():
    """Klasa zawierająca testy, które będą wykonywane na wskazanych grupach badanych"""    
    def __init__(self, proba):
        self.proba = proba
        
   
    def histogram(self):
        #sprawdzanie histogramu
        #wybór odpowiedniej ścieki
        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2', self.proba)
        os.chdir(sciezka)
        extension = 'csv'
        katalog = self.proba + ".csv"
        #sprawdza czy plik csv istnieje, jeśli tak od razu tworzy hsitogram


        import re
        tekst = self.proba
        wzorzec = re.compile(r'\W(\w+)\W(\w+)\W(\w+)\W(\w+)\W(\w+)')
        print(wzorzec.sub(r'\5', tekst))
        tytul = wzorzec.sub(r'\5', tekst)

        if os.path.exists(katalog):
            dane = pd.read_csv(self.proba + '.csv')
            fraktal = dane[dane['Area'].isin(['0'])]
            df = pd.DataFrame(fraktal)
            plt.hist(df['D'], bins=50, alpha=0.5, label='D')
            #plt.legend(loc='upper right')
            
            
            plt.xlabel('Wymiar fraktalny',fontsize=10)
            plt.ylabel('Liczba komórek',fontsize=10)
            
            plt.title(label= tytul + " - histogram", fontsize=12, color="blue")
            
        #jeśli nie istnieje - tworzy plik, a potem na jego podstawie histogram 
        else:
            pliki = [i for i in glob.glob('*.{}'.format(extension))]
            polaczone = pd.concat([pd.read_csv(f) for f in pliki ])
            polaczone.to_csv( self.proba + ".csv", index=False, encoding='utf-8-sig')
            #składanie wierszy z analizy fraktalnej
            dane = pd.read_csv(self.proba + '.csv')
            fraktal = dane[dane['Area'].isin(['0'])]
            df = pd.DataFrame(fraktal)     
            plt.hist(df['D'], bins=50, alpha=0.5, label='D')
            #plt.legend(loc='upper right')

            plt.xlabel('Wymiar fraktalny',fontsize=10)
            plt.ylabel('Liczba komórek',fontsize=10)

            plt.title(label= tytul + " - histogram", fontsize=12, color="blue")
            

    def qq_wykres(self):

        #tworzy wykres kwantyl-kwantyl 
        #wybór ścieki
        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2', self.proba)
        os.chdir(sciezka)
        extension = 'csv'

        katalog = self.proba + ".csv"


        import re
        tekst = self.proba
        wzorzec = re.compile(r'\W(\w+)\W(\w+)\W(\w+)\W(\w+)\W(\w+)')
        print(wzorzec.sub(r'\5', tekst))
        tytul = wzorzec.sub(r'\5', tekst)

        #sprawdzenie czy plik csv istnieje - jeśli tak tworzy wykres (z dodatkowego modułu stat)
        if os.path.exists(katalog):
            dane = pd.read_csv(self.proba + '.csv')
            fraktal = dane[dane['Area'].isin(['0'])]
            df = pd.DataFrame(fraktal)
            wykres = stats.probplot(df['D'], dist="norm", plot=pylab)

            plt.xlabel('Kwantyle teoretyczne. Normalny rozkład kwantylowy',fontsize=10)
            plt.ylabel('Kwantyle z próby',fontsize=10)

            plt.title(label= tytul + " - wykres kwantyl-kwantyl", fontsize=12, color="green")
            
        else:
            pliki = [i for i in glob.glob('*.{}'.format(extension))]
            polaczone = pd.concat([pd.read_csv(f) for f in pliki ])
            polaczone.to_csv( self.proba + ".csv", index=False, encoding='utf-8-sig')
            #składanie wierszy z analizy fraktalnej
            dane = pd.read_csv(self.proba + '.csv')
            fraktal = dane[dane['Area'].isin(['0'])]
            
            df = pd.DataFrame(fraktal)
            wykres = stats.probplot(df['D'], dist="norm", plot=pylab)
            
            plt.xlabel('Kwantyle teoretyczne. Normalny rozkład kwantylowy',fontsize=10)
            plt.ylabel('Kwantyle z próby',fontsize=10)

            plt.title(label= tytul + " - wykres kwantyl-kwantyl", fontsize=12, color="green")
            
        
    def test_shapiro_wilka(self):
        #przeprowadzenie testu shapiro-wilka te z pomocą modułu stat
        katalog = os.path.expanduser('~')
        sciezka = os.path.join(katalog, 'fraktal_2', self.proba)
        os.chdir(sciezka)
        extension = 'csv'

        katalog = self.proba + ".csv"
        #sprawdzenie czy plik istnieje
        if os.path.exists(katalog):
            dane = pd.read_csv(self.proba + '.csv')
            fraktal = dane[dane['Area'].isin(['0'])]

            df = pd.DataFrame(fraktal)
            shapiro(df['D'])
            stat, p = shapiro(df['D'])
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            #wybrany poziom istotności
            alpha = 0.05
            #zwraca wynik
            if p > alpha:
                print(self.proba + ' : Rozkład przypomina rozkład Gaussa (nie można odrzucic H0)')
            else:
                print(self.proba +' : Rozkład nie przypomina rozkładu Gaussa (należy odrzucić H0)')

        else:
            pliki = [i for i in glob.glob('*.{}'.format(extension))]
            polaczone = pd.concat([pd.read_csv(f) for f in pliki ])
            polaczone.to_csv( self.proba + ".csv", index=False, encoding='utf-8-sig')
            #składanie wierszy z analizy fraktalnej
            dane = pd.read_csv(self.proba + '.csv')
            fraktal = dane[dane['Area'].isin(['0'])]
            
            df = pd.DataFrame(fraktal)

            shapiro(df['D'])
            stat, p = shapiro(df['D'])
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            #poziom istotności
            alpha = 0.05
            #zwraca wynik
            if p > alpha:
                print(self.proba + ' : Rozkład przypomina rozkład Gaussa (nie można odrzucic H0)')
                return 'przypomina'
            else:
                print(self.proba +' : Rozkład nie przypomina rozkładu Gaussa (należy odrzucić H0)')
                return 'nie_przypomina'