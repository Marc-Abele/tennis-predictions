import pandas as pd
import matplotlib.pyplot as plt
import pickle
import openpyxl

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

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
        
class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clustering_model=None):
        self.clustering_model = clustering_model
        self.clusters_ = None
        self.X_clustered = None

    def fit(self, X, y=None):
        self.clustering_model.fit(X)
        self.clusters_ = self.clustering_model.predict(X)
        self.X_clustered = pd.DataFrame(X)
        self.X_clustered['Cluster'] = self.clusters_
        return self

    def transform(self, X):
        return self.X_clustered
    
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
        df_pred=pd.read_csv(self.df_pred_file)
        try:
            df_pred['SumOdd']=df_pred['SumOdd'].apply(lambda x : x.replace(',', '.'))
            df_pred['GapOdd']=df_pred['GapOdd'].apply(lambda x : x.replace(',', '.'))
        except:
            pass
        df_pred['SumOdd']=df_pred['SumOdd'].astype('float')
        df_pred['GapOdd']=df_pred['GapOdd'].astype('float')
        self.df_pred=df_pred
        
        self.df_real=pd.read_csv(self.df_real_file)
        print('dfs read...')

    def load_algorithms(self):
        # Charger les algorithmes de prédiction pour chaque type de prédiction
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
        s=self.algo_nbgames.predict(self.df_pred[['Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'GapRank', 'GapOdd', 'Month', 'SumOdd', 'SumRank']]).round()
        col='pred_nbgames'
        self.df_pred[col]=s

    def save_predictions(self):
        new_df=pd.concat([self.df_real, self.df_pred]).reset_index(drop=True)
        new_df.to_csv('real.csv', index=False)
        self.df_pred.to_csv('pred.csv', index=False)
            
class Predictor_ATP_5sets:
    def __init__(self):
        historical_data='atp.csv'
        self.historical_data=pd.read_csv(historical_data)
        self.circuit = 'ATP'
        self.type_of_match = '3 winning sets'
        self.df_pred_file = 'pred_5sets.csv'
        self.df_real_file = 'real_5sets.csv'
        self.pipeline_nbsets = 'pipeline_atp_5sets_nbsets.pkl'
        self.pipeline_nbgames = 'pipeline_atp_5sets_nbgames.pkl'
        self.df_pred = None
        self.load_data()
        self.load_algorithms()

    def load_data(self):
        df_pred=pd.read_csv(self.df_pred_file)
        try:
            df_pred['SumOdd']=df_pred['SumOdd'].apply(lambda x : x.replace(',', '.'))
            df_pred['GapOdd']=df_pred['GapOdd'].apply(lambda x : x.replace(',', '.'))
        except:
            pass
        df_pred['SumOdd']=df_pred['SumOdd'].astype('float')
        df_pred['GapOdd']=df_pred['GapOdd'].astype('float')
        self.df_pred=df_pred
        
        self.df_real=pd.read_csv(self.df_real_file)
        print('dfs read...')

    def load_algorithms(self):
        # Charger les algorithmes de prédiction pour chaque type de prédiction
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
        s=self.algo_nbgames.predict(self.df_pred[['Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'GapRank', 'GapOdd', 'Month', 'SumOdd', 'SumRank']]).round()
        col='pred_nbgames'
        self.df_pred[col]=s

    def save_predictions(self):
        new_df=pd.concat([self.df_real, self.df_pred]).reset_index(drop=True)
        new_df.to_csv('real_5sets.csv', index=False)
        self.df_pred.to_csv('pred_5sets.csv', index=False)
        
class Predictor_WTA:
    def __init__(self):
        historical_data='wta.csv'
        self.historical_data=pd.read_csv(historical_data)
        self.circuit = 'WTA'
        self.type_of_match = '2 winning sets'
        self.df_pred_file = 'pred_3sets_wta.csv'
        self.df_real_file = 'real_3sets_wta.csv'
        self.pipeline_nbsets = 'pipeline_wta_nbsets.pkl'
        self.pipeline_nbgames = 'pipeline_wta_nbgames.pkl'
        self.df_pred = None
        self.load_data()
        self.load_algorithms()

    def load_algorithms(self):
        # Charger les algorithmes de prédiction pour chaque type de prédiction
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
        s=self.algo_nbsets.predict(self.df_pred[['Location', 'Tournament', 'Tier', 'Court', 'Surface', 'Round', 'GapRank', 'GapOdd', 'Month', 'SumOdd', 'SumRank']])
        col='pred_nbsets'
        self.df_pred[col]=s

    def prediction_games(self):
        # Effectuer la prédiction du nombre de jeux pour le match en utilisant self.algo_games
        # Enregistrer les résultats dans self.df_pred
        s=self.algo_nbgames.predict(self.df_pred[['Location', 'Tournament', 'Tier', 'Court', 'Surface', 'Round', 'GapRank', 'GapOdd', 'Month', 'SumOdd', 'SumRank']]).round()
        col='pred_nbgames'
        self.df_pred[col]=s

    def save_predictions(self):
        new_df=pd.concat([self.df_real, self.df_pred]).reset_index(drop=True)
        new_df.to_csv('real_wta.csv', index=False)
        self.df_pred.to_csv('pred_wta.csv', index=False)