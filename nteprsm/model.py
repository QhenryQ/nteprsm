# Imports for data pre-processing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
# Imports for stan tools
# from gptools.stan import compile_model # cmdstanpy has a complie model function, plus this repo is not actively maintained
import cmdstanpy
import nest_asyncio
# Imports for save/load
import pickle
import datetime
# Imports for data visualization and posterior chain analysis
import arviz as az
import plotly.express as px
import plotly.graph_objects as go
import random

from settings import MODEL_DIR, DATA_DIR, SRC_DIR

class NTEPModel:
    def __init__(self, load_file :str = None,
                    padding: int = 5, pred_N: int = 100, num_basis_functions: int = 8,
                    stan_file :str= 'model_01-feb-2024.stan',
                    data_file :str= 'quality_nj2.csv',
                    
                    model_folder: str= str(MODEL_DIR) + "/",
                    data_folder: str= str(DATA_DIR) + "/",
                    src_folder: str= str(SRC_DIR) + "/",
                    ):
        self.stan_file = stan_file
        self.data_file = data_file
        
        self.model_folder = model_folder
        self.data_folder = data_folder
        self.src_folder = src_folder
        
        self.padding = padding
        self.pred_N = pred_N
        self.num_basis_functions = num_basis_functions
        # data is serialized into .pkl
        self.df = None
        self.data = None
        self.model = None
        self.fit = None
        self.az_data = None
        if load_file:
            self.load(load_file)
        elif data_file:
            self.get_data()
        #elif data_path:
        #    self.get_data()
        #    if stan_path: self.sample()
    def get_data(self, file: str = None):
        if file == None: file = self.data_file
        df = pd.read_csv(self.data_folder+file)
        encoder1, encoder2, encoder3, encoder4 = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
        df['ENTRY_NAME_CODE'] = encoder2.fit_transform(df['ENTRY_NAME'])
        df['PLT_ID_CODE'] = encoder3.fit_transform(df['PLT_ID'])
        df['RATER_CODE'] = encoder4.fit_transform(df['RATER'])
        plt_coords = df.groupby('PLT_ID_CODE')[['ROW','COL']].mean()
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear-(df['DATE'].dt.is_leap_year*(df['DATE']).dt.dayofyear >= 60)
        df['TIME_OF_YEAR'] = df['DAY_OF_YEAR']/365
        #num_days_in_year = pd.to_datetime(year.astype(str), format='%Y').dt.is_leap_year * 366 + (~pd.to_datetime(year.astype(str), format='%Y').dt.is_leap_year) * 365
        #df['TIME_OF_YEAR'] = pd.to_datetime(df['DATE']).dt.dayofyear/num_days_in_year
        df['ENTRY_CUMCOUNT'] = df.groupby('ENTRY_NAME').cumcount() + 1
        
        ##### data for old model #####
        df['RATING_EVENT_CODE'] = encoder1.fit_transform(df['RATING_EVENT']) # For comparison with the old model
        coords = np.array([plt_coords['ROW'].values, plt_coords['COL'].values])
        distances = cdist(coords.T,coords.T, metric='euclidean')
        ##############################
        self.df = df
        self.data = {"y": df["QUALITY"].values-1,                       # target variable, the rating value we try to model
        
                     # general data
                     "N": len(df["QUALITY"]),                           # Number of responses
                     "num_raters":len(df['RATER'].unique()),            # Total number of rating events
                     "num_entries":len(df['ENTRY_NAME'].unique()),      # Total number of entries (turfgrass types)
                     "num_plots":len(df['PLT_ID'].unique()),            # Total number of plots 
                     "num_categories": 9,                               # Total number of rating categories
                     "rater_id": df["RATER_CODE"].values+1,             # rating id for y[n], defined by rater + date
                     "entry_id": df["ENTRY_NAME_CODE"].values+1,        # entry of y[n]
                     "plot_id": df["PLT_ID_CODE"].values+1,             # plot id of y[n]
                    
                     # values used for fourier scalable gaussian process, plot effect
                     "num_ratings_per_entry": np.max(df.groupby('ENTRY_NAME').count()['PLT_ID']),# number of ratings each entry received
                     "num_rows": int (plt_coords["ROW"].unique().max()),                        # number of rows of the turf plot grid
                     "num_cols": int (plt_coords["COL"].unique().max()),                        # number of cols of the turf plot grid
                     "num_rows_padded": int (plt_coords["ROW"].unique().max()) + self.padding,  # number of rows + padding
                     "num_cols_padded": int (plt_coords["COL"].unique().max()) + self.padding,  # number of columns + padding  
                     "plot_row" : plt_coords["ROW"].astype(int),                                # row of the plot corresponding to plot_id
                     "plot_col" : plt_coords["COL"].astype(int),                                # column of the plot corresponding to plot_id
                          
                     # values used for hilbert basis scalable gaussian process, time effect
                     "time" : df['TIME_OF_YEAR'],               # From 0 to 1, based on day of year (we pretend Feb 29 doesnt exist and move it to Feb 28)
                     "M_f":self.num_basis_functions,            # number of Hilbert Basis functions
                     "entry_cumcount": df["ENTRY_CUMCOUNT"],    # cumulative count of the entry, e.g. if it is the 2nd time it appeared as an entry, it would be "2"
                     
                     # values for making predictions
                     "pred_N": self.pred_N,
                     "pred_time": np.linspace(0,1,self.pred_N+1)[1:],
                     
                     # data used only by the old model
                     "num_rating_events":len(df['RATING_EVENT'].unique()),
                     "rating_event_id":df["RATING_EVENT_CODE"].values+1,
                     "plot_distance":distances,
                      }
        self.data_file = file
        print("Loaded data from "+self.data_folder+file)
    # Returns a dictionary of  {entry_code (str) : entry_code (integer)} pairs
    # if invert=True, returns a dictionary of {entry_code (integer) : entry_code (str)} pairs
    def get_entry_codes(self, invert=False):
        entry_codes = dict(self.df.groupby('ENTRY_NAME')['ENTRY_NAME_CODE'].mean().apply(lambda x: round(x)))
        if invert:
            inv_entry_codes = {}
            for key in entry_codes.keys(): inv_entry_codes[entry_codes[key]] = key
            return inv_entry_codes
        return entry_codes
    # Returns a dictionary of  {entry_code (str) : entry_code (integer)} pairs
    # if invert=True, returns a dictionary of {entry_code (integer) : entry_code (str)} pairs
    def get_rater_codes(self, invert=False):
        rater_codes = dict(m.df.groupby('RATER')['RATER_CODE'].mean().apply(lambda x: round(x)))
        if invert:
            inv_rater_codes = {}
            for key in rater_codes.keys(): inv_rater_codes[rater_codes[key]] = key
            return inv_rater_codes
        return rater_codes
    # Run MCMC sampling to fit the model in stan.
    # Custom models can be fitted, but not all features are avaliable in a custom model!
    def sample(self, file: str = None):
        if file == None: file = self.stan_file
        cmdstanpy.install_cmdstan()  # check existence first
        nest_asyncio.apply()
        print("Running MCMC chains for model : "+self.src_folder+file)
        if file: self.model = compile_model(stan_file=self.src_folder+file, force_compile=True)  # this function is from a package which is not actively maintained, how about https://cmdstanpy.readthedocs.io/en/stable-0.9.65/api.html?highlight=CmdStanModel#cmdstanpy.CmdStanModel.compile:~:text=compile(force,%5Bsource%5D
        self.fit = self.model.sample(self.data)
        self.stan_file = file
        self.az_data = az.from_cmdstanpy(
            posterior=self.fit,
            observed_data={"y": self.data['y']},
            log_likelihood="log_lik",
        )
    def save(self, file: str=None):
        if file == None: file = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")+".pkl"
        if self.fit:
            with open(self.model_folder+file, "wb") as f:
                pickle.dump({'model' : self.model,
                            'fit' : self.fit,
                            'data' : self.data,
                            'data_file':self.data_file, ## some old test versions used "data_path" instead of "data_file"
                            'stan_file':self.stan_file,
                            'model_folder':self.model_folder,
                            'src_folder':self.src_folder,
                            'data_folder':self.data_folder,
                            #'df' : self.df,
                            'padding':self.padding,
                            'pred_N':self.pred_N,
                            'num_basis_functions':self.num_basis_functions}, f, protocol=-1)
            print("Saved to "+self.model_folder+file)
        else:
            print("No fit data found.")
    def load(self, file: str):
        with open(self.model_folder + file, "rb") as f_pickle:
            f = pickle.load(f_pickle)
            keys = f.keys()
            #if 'data' in keys: self.data = f['data']
            #else: print("Warning: Data not found in file!")
            if 'model' in keys: self.model = f['model']
            else: print("Warning: Model not found in file!")
            if 'fit' in keys: self.fit = f['fit']
            else: print("Warning: Fit not found in file!")
            if 'stan_file' in keys: self.stan_file = f['stan_file']
            else: print("Warning: stan file location not found in file!")
            if 'padding' in keys: self.padding = f['padding']
            else: print("Warning: padding not found in file!")
            if 'pred_N' in keys: self.pred_N = f['pred_N']
            else: print("Warning: pred_N not found in file!")
            if 'num_basis_functions' in keys: self.num_basis_functions = f['num_basis_functions']
            else: print("Warning: num_basis_functions not found in file!")
            if 'model_folder' in keys: self.model_folder = f['model_folder']
            else: print("Warning: model folder location not found in file!")
            if 'data_folder' in keys: self.data_folder = f['data_folder']
            else: print("Warning: data folder location not found in file!")
            if 'src_folder' in keys: self.src_folder = f['src_folder']
            else: print("Warning: src folder location not found in file!")
            if 'data_file' in keys:
                self.data_file = f['data_file']
                if 'data_folder' in keys: self.get_data()
            else: print("Warning: data file location not found in file!")
        self.az_data = az.from_cmdstanpy(
            posterior=self.fit,
            observed_data={"y": self.data['y']},
            log_likelihood="log_lik",
        )
        print("Loaded model from "+self.model_folder+file)
    # TODO : Adjust default argument list
    # Useful for checking for convergence issues and the distribution of some variables
    def plot_trace(self,var_names: list =['sigma_plot','lengthscale_plot','sigma_f','lengthscale_f','sigma_tau_rater']):
        for variable in var_names:
            try: az.plot_trace(self.az_data, compact=False, var_names=variable)
            except: pass
    def loo(self):
        return az.loo(self.az_data)
    ## returns a list of scores for each entry
    # accepted values for scoring: 'weighted','annual','data'
    ## TODO: Add month breakdowns?
    def get_entry_scores(self, scoring: str = 'weighted', months: list = []):
        month_vals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']  #consider save them to constants
        # if months is not all?
        if len(months) != 0:
            x_pred = [0]+list((self.data['pred_time'][:-1])*365)
        
        entry_means = np.array([self.fit.pred_time_effect[:,i].mean(axis=0) for i in range(self.data["num_entries"])])
        if scoring == 'weighted':
            entry_std = self.fit.pred_time_effect.std(axis=0).mean(axis=0)
            entry_std_min = entry_std.min()
            entry_std_max = entry_std.max()
            weights = 0.75*(entry_std_max - entry_std)/(entry_std_max-entry_std_min) + 0.25
            return entry_means@weights          # adjust based on scoring method
        elif scoring == 'annual': return entry_means.mean(axis=1)
        elif scoring == 'data': return self.fit.time_effect.mean(axis=0).mean(axis=1)
        else:
            print("Warning: "+str(scoring)+"is not an accepted scoring method. Defaulting to weighted scoring.")
            return self.get_entry_scores(scoring='weighted')
    ## entries can either be a list of entries we want to plot, or an integer equal to the number of random entries      
    def plot_time_effect(self, entries = 26, # can be an int or a list
                                colors = px.colors.qualitative.Dark24,  # plotly colors
                                credit_interval = None,             # None or a float
                                sort_entries: str = 'weighted',  # accepts:  'unsorted', 'weighted', 'annual', 'data', might need better names for these
                                dimensions = None,                      # (Optional) None or a tuple (width, height)                 
                                ):
                                
        ## Convert strings entries to their integer counterparts
        entry_codes = self.get_entry_codes()
        inv_entry_codes = self.get_entry_codes(invert=True)
        if isinstance(entries,list):
            new_entries = []
            for i in entries:
                if isinstance(i,int) and i >= 0 and i < self.data["num_entries"]: new_entries.append(i)
                elif isinstance(i,str) and i in entry_codes.keys(): new_entries.append(entry_codes[i])
                else: print("Warning: "+str(i)+" is an invalid entry!")
            # remove duplicates
            entries = list(dict.fromkeys(new_entries))
        
        month_vals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # handle all entries (will feed into the random entries)
        if isinstance(entries,str) and entries.lower() == 'all':
            entries = self.data["num_entries"]
            print("Loading all entries...")
        
        # Handle random entries
        if isinstance(entries,int):
            all_entries = [i for i in range(self.data["num_entries"])]
            random.shuffle(all_entries)
            entries = all_entries[:entries]
        entry_means = np.array([self.fit.pred_time_effect[:,i].mean(axis=0) for i in range(self.data["num_entries"])])
        
        # Handle sorting entries
        if sort_entries != 'unsorted':
            entry_scores = self.get_entry_scores(scoring=sort_entries)
            entries = sorted(entries, key = lambda x: -1*entry_scores[x])
        # calculating credit interval lines
        if credit_interval:
            q_low, q_high = (1-credit_interval)/2, (1+credit_interval)/2
            pred_low = np.array([[np.quantile(self.fit.pred_time_effect[:,i,j],q_low) for i in entries] for j in range(self.fit.pred_time_effect.shape[2])])
            pred_high = np.array([[np.quantile(self.fit.pred_time_effect[:,i,j],q_high) for i in entries] for j in range(self.fit.pred_time_effect.shape[2])])
        fig = go.Figure()
        
        # prepare variables and plot
        df = self.df
        for idx, i in enumerate(entries):
            x = list((df[df["ENTRY_NAME_CODE"] == i]['TIME_OF_YEAR']*365))
            sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
            x_sorted = [list((df[df["ENTRY_NAME_CODE"] == i]['TIME_OF_YEAR']*365))[ind] for ind in sorted_indices]
            y_sorted = [self.fit.time_effect[:,i].mean(axis=0)[ind] for ind in sorted_indices]    
            entry_name = inv_entry_codes[i]
            # plotting values of data points
            fig.add_trace(go.Scatter(x=x_sorted, y=y_sorted, mode='markers', marker=dict(size=5, color=colors[idx%len(colors)]),
                                     name=entry_name, legendgroup=entry_name))
            # wrapping around to make sure the 0th element is 0
            x_pred = (self.data['pred_time'])*365
            y_pred = self.fit.pred_time_effect[:,i].mean(axis=0)
            # plotting prediction mean line
            fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode='lines', line=dict(width=1.5, color=colors[idx%len(colors)]),
                                     name=entry_name, legendgroup=entry_name, showlegend=False, hoverinfo='none'))
            if credit_interval:
            
                y_pred_low = pred_low[:,idx]
                y_pred_high = pred_high[:,idx]
                fig.add_trace(go.Scatter(x=x_pred, y=y_pred_low, mode='lines', line=dict(width=0.5, color=colors[idx%len(colors)]),
                                         name=entry_name, legendgroup=entry_name, showlegend=False, hoverinfo='none'))
                fig.add_trace(go.Scatter(x=x_pred, y=y_pred_high, mode='lines', line=dict(width=0.5, color=colors[idx%len(colors)]),
                                         name=entry_name, legendgroup=entry_name, showlegend=False, fill='tonexty', hoverinfo='none'))
        fig.update_layout(title='Mean Time Effect',
                          xaxis=dict(
                            tickmode='array',
                            tickvals=month_vals,
                            ticktext=month_labels,
                            tickfont=dict(size=18)
                          ),
                          yaxis_title='Effect',
                          yaxis=dict(
                            title_font=dict(size=20)
                          ),
                          legend=dict(
                            font=dict(size=16)  # Increase the font size for the legend
                          ),
                          title_font=dict(size=24)
                          )
        if dimensions: fig.update_layout(width=dimensions[0], height=dimensions[1])
        if credit_interval:  fig.update_layout(title='Time Effect on Turfgrass Quality Ratings, '+str(round(credit_interval*100,2))+"% Credit Interval")
        #if file_path: fig.write_html(file_path)
        return fig
    ## Thoughts: 
    ## Mouse-over information, what to do with it? Plotly interactive mouse-overs for heatmaps not powerful
    def plot_plot_effect(self, height=650, width=650):
        df = self.df
        PLT_ROW = df.groupby('PLT_ID')[['ROW', 'COL']].max()['ROW']
        PLT_COL = df.groupby('PLT_ID')[['ROW', 'COL']].max()['COL']
        
        inv_entry_codes = self.get_entry_codes(invert=True)
        PLT_ENTRY_NAME = df.groupby('PLT_ID')[['ENTRY_NAME_CODE']].max()['ENTRY_NAME_CODE'].apply(lambda x : inv_entry_codes[x])

        mean_matrix_time = np.array([[np.nan for i in range(np.max(PLT_COL))] for j in range(np.max(PLT_ROW))])
        mean_matrix_time[PLT_ROW-1, PLT_COL-1] = self.fit.plot_effect.mean(axis=0)
        plot_entry_names = np.array([[None for i in range(np.max(PLT_COL))] for j in range(np.max(PLT_ROW))])
        plot_entry_names[PLT_ROW-1, PLT_COL-1] = PLT_ENTRY_NAME
        fig = go.Figure(data=go.Heatmap(
            z=mean_matrix_time,
            text=plot_entry_names,
            hovertemplate='Entry: %{text}<br>Effect: %{z}<extra></extra>',  # Define the hover template to display the custom strings
            colorscale='Viridis',
            colorbar=dict(thickness=30,
                  tickfont=dict(size=16))
        ))
        fig.update_xaxes(tickmode='array',
                         tickvals=[i for i in range(np.max(PLT_COL))],
                         ticktext=[str(i+1) for i in range(np.max(PLT_COL))],
                         tickfont=dict(size=16)
                        )
        fig.update_yaxes(tickmode='array',
                         tickvals=[i for i in range(np.max(PLT_ROW))],
                         ticktext=[str(i+1) for i in range(np.max(PLT_ROW))],
                         tickfont=dict(size=16)
                        )
        fig.update_layout(
            title='Mean Plot Effect',
            width=width,  # Set the width to 800 pixels
            height=height,  # Set the height to 600 pixels
            title_font=dict(size=22),
        )
        fig.show()
    def plot_tau(self):
        pass
    # score entries
    # adjust for seasons? 
    def plot_entry_scores(entries='all', scoring_method: str ='weighted', season='all'):
        pass
