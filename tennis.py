import pandas as pd
import matplotlib.pyplot as plt
import pickle
import openpyxl

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

class Tennis:
    '''
    Instanciates a Tennis object
    :param str circuit: atp or wta
    :param str type_of_match: 3sets or 5sets    
    '''
    def __init__(self, circuit, type_of_match):#, target):
        self.circuit=circuit
        self.type_of_match=type_of_match
        #self.target=target
        
        csv_file=str(circuit)+'_'+str(type_of_match)+'.csv'
        self.csv_file=csv_file
        self.df=pd.read_csv(csv_file)
        self.stats=Stats(self.df)
    

class Stats:
    def __init__(self, df):
        self.df_number=df.select_dtypes(include='number')
        self.df_cat=df.select_dtypes(include='object')
    
    def hist_boxplot(self):
        '''
        returns an histogram & a boxplot by row
        iterating on numerical columns of the original dataframe
        '''
        num_cols = len(self.df_number.columns)
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, num_cols*3))
        for i, col in enumerate(self.df_number.columns):
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
            ax1.hist(self.df_number[col])
            ax1.set_title(col + ' histogram')
            
            ax2.boxplot(self.df_number[col])
            ax2.set_title(col + ' boxplot') 
        plt.tight_layout()
        plt.show()
        
    def multiple_hist(self):
        '''
        returns the histograms of numerical columns of the original dataframe
        '''
        plt.figure(figsize=(18,25))
        for i in range(len(self.df_number.columns)):
            plt.subplot(8, 5, i + 1)    # define figure size and subplots numbers
            plt.hist(x = self.df_number[self.df_number.columns[i]])
            plt.title(self.df_number.columns[i] + ' HISTOGRAM')
        plt.show()
        
    def multiple_boxplot(self):
        '''
        returns the boxplots of numerical columns of the original dataframe
        '''
        plt.figure(figsize=(18,25))
        for i in range(len(self.df_number.columns)):
            plt.subplot(8, 5, i + 1)    # define figure size and subplots numbers
            plt.boxplot(x = self.df_number[self.df_number.columns[i]])
            plt.title(self.df_number.columns[i] + ' BOXPLOT')
        plt.show()

    
"""class Tennis_3sets(Tennis):
    def __init__(self, *args, **kwargs):
        '''
        Instanciates Tennis_3sets object, inherited from Tennis
        :param csv_file str: name of the csv file with historical datas with 2 winning sets matches
        '''
        Tennis.__init__(self, *args, **kwargs)
        self.description='Matchs en 2 sets gagnants'
""" 
"""class Tennis_5sets(Tennis):
    def __init__(self, *args, **kwargs):
        '''
        Instanciates Tennis_5sets object, inherited from Tennis
        :param csv_file str: name of the csv file with historical datas with 3 winning sets matches
        '''
        Tennis.__init__(self, *args, **kwargs)
        self.description='Matchs en 3 sets gagnants'
"""            
"""class Predictor(Tennis):
    '''
    Instanciates a Predictor class
    :param target str: the target you want to predict
    
    For now, the only target we are able to predict is the total number of games in a match
    '''+ Tennis.__doc__
    
    def __init__(self, target, *args, **kwargs):
        self.target=target
        self._tennis=Tennis(*args, **kwargs)
        self.df_pred_file='results.xlsx'
        self.df_pred=None
        self.df_real=None
        self.read_df()
        pipeline_file='pipeline_'+self._tennis.circuit+'_'+self._tennis.type_of_match+'_'+self.target+'.pkl'
        self.pipeline_file=pipeline_file
        self.load_pipeline()
        
        print('Ready to predict !')
        
    def read_df(self):
        self.df_pred=pd.read_excel(self.df_pred_file, sheet_name='pred')
        self.df_real=pd.read_excel(self.df_pred_file, sheet_name='real')
        print('dfs read...')
        
    def load_pipeline(self):
        # only works with scikit-learn 1.0.2, not with scikit-learn 1.2.2
        with open(self.pipeline_file, 'rb') as fichier:
            self.algo = pickle.load(fichier)
        print('algo loaded...')

    
    def prediction(self):
        s=self.algo.predict(self.df_pred[['Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'GapRank', 'GapOdd', 'Month']])
        col='pred_'+self.target
        self.df_pred[col]=s
        return self.df_pred
    
    def add_to_real(self):
        self.df_real=pd.concat([self.df_real, self.df_pred])
        print('Concatenation OK')
        return self.df_real

    def save_all(self):
        workbook = openpyxl.Workbook()
        writer = pd.ExcelWriter('results.xlsx', engine='openpyxl')
        writer.book = workbook

        self.df_real.to_excel(writer, sheet_name='real', index=False)
        self.df_pred.to_excel(writer, sheet_name='pred', index=False)
        
        # Supprimer l'onglet par défaut "Sheet"
        del workbook['Sheet']
            
        writer.save()
        writer.close()
"""


class Predictor_ATP_3sets:
    def __init__(self):
        historical_data='atp.csv'
        self.historical_data=pd.read_csv(historical_data)
        self.circuit = 'ATP'
        self.type_of_match = '2 winning sets'
        self.df_pred_file = 'pred.csv'
        self.df_real_file = 'real.csv'
        self.pipeline_nbsets = 'pipeline_atp_3sets_nbsets.pkl'
        self.pipeline_nbgames = 'pipeline_atp_3sets_nbgames.pkl'
        self.df_pred = None
        self.load_data()
        self.load_algorithms()

    def load_data(self):
        # Charger les données nécessaires (ex : fichier results.xlsx)
        self.df_pred=pd.read_csv(self.df_pred_file)
        self.df_real=pd.read_csv(self.df_real_file)
        print('dfs read...')

    def load_algorithms(self):
        # Charger les algorithmes de prédiction pour chaque type de prédiction
        # et les stocker dans des attributs de la classe (ex : self.algo_winner, self.algo_sets, self.algo_games)
        # only works with scikit-learn 1.0.2, not with scikit-learn 1.2.2
        with open(self.pipeline_nbsets, 'rb') as fichier1:
            self.algo_nbsets = pickle.load(fichier1)
        with open(self.pipeline_nbgames, 'rb') as fichier2:
            self.algo_nbgames = pickle.load(fichier2)
        print('algo loaded...')

    def perform_predictions(self):
        # Effectuer toutes les prédictions pour le match en utilisant les algorithmes correspondants
        self.prediction_sets()
        self.prediction_games()
        return self.df_pred

    def prediction_sets(self):
        # Effectuer la prédiction du nombre de sets pour le match en utilisant self.algo_sets
        # Enregistrer les résultats dans self.df_pred
        s=self.algo_nbsets.predict(self.df_pred[['Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'GapRank', 'GapOdd', 'Month', 'SumOdd', 'SumRank']])
        col='pred_nbsets'
        self.df_pred[col]=s

    def prediction_games(self):
        # Effectuer la prédiction du nombre de jeux pour le match en utilisant self.algo_games
        # Enregistrer les résultats dans self.df_pred
        s=self.algo_nbgames.predict(self.df_pred[['Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'GapRank', 'GapOdd', 'Month']])
        col='pred_nbgames'
        self.df_pred[col]=s

    def save_predictions(self):
        new_df=pd.concat([self.df_real, self.df_pred]).reset_index(drop=True)
        new_df.to_csv('real.csv', index=False)
        self.df_pred.to_csv('pred.csv', index=False)