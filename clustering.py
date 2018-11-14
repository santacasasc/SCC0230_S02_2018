'''
This script perfoms the basic process for applying a machine learning
Algorithm for a set of data using Python libraries.

The five steps are:
   1. Read a data set (using Pandas)
   2. Transform and filter data (using Pandas)
   3. Process the numerical data (using NumPy)
   4. Train and evaluate apprentices (using Scikit-Learn)
   5. Plot and compare results (using matplotlib)

============
Dataset
============
Considerations:
    The data were previously extracted in order to anonymize the patients present

File fields:
    "CD_ATENDIMENTO_INDEX"          Index to identify a person who passed the institution
    "TP_ATENDIMENTO"                Index to indicate the type of service 
    "CD_SGRU_CID"                   Index of diagnostic group 
    "DS_SGRU_CID"                   Description of diagnostic group 
    "CD_CID"                        Index of diagnostic
    "DS_CID"                        Description of diagnostic 
    "CD_PRE_MED_INDEX"              Index to identify a prescrition   
    "NM_OBJETO"                     Index to identify the prescription type
    "CD_PACIENTE_INDEX"             Index to identify a person
    "NR_DIAS_NO_ATENDIMENTO"        Age in patient days on the day of attendance
    "TP_SEXO"                       Index to identify a person sex 
    "NR_DIAS_PRESCIACAO"            Number of days of percretion after service           
    "CD_TIP_ESQ"                    Index of Prescription item group  
    "DS_TIP_ESQ"                    Description of Prescription item group 
    "CD_TIP_PRESC_INDEX"            Index of Prescription item 
    "DS_TIP_PRESC"                  Description of Prescription item 

'''

# File location of dataset
File = ".\\dataset\\base_ia_v1.zip"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def clear_old():
    '''
    Clear all old results.
    '''
    import os
    import shutil
    folder = '.\\results'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

# =====================================================================

def read_xlsx_data():
    '''
    Read a xlsx data for this script into a pandas DataFrame.
    '''

    xlsx = pd.ExcelFile(File)

    farame_sheets = []

    for sheet in xlsx.sheet_names:
        farame_sheets.append(xlsx.parse(sheet))

    return pd.concat(farame_sheets, sort=False)

# =====================================================================

def read_csv_data():
    '''
    Read a csv data for this script into a pandas DataFrame.
    '''

    return pd.read_csv(File,encoding  = 'windows-1252',compression = 'zip')

# =====================================================================

def get_nm_objeto_list(frame):
    '''
    Search for different types of prescriptions and cleaning of invalid types of nm_objetos.
    '''

    list_filter = list()
    list_filter.append('PROCEDIMENTO AIH') # Prescription for generating document, does not contain relevant information
    list_filter.append('TRANSCRIÇÃO MÉDICA') # Nursing transcription does not contain relevant information for this analisys 
    list_filter.append('SOLICITAÇÃO MATERAIS') # Used rarely, do not contain relevant information.
    list_filter.append('PRESCRIÇÃO FISIOTERAPIA')  # only two prescription items
    list_filter.append('PRESCRIÇÃO HEMOTERAPIA') # Prescription for blood request generation, does not contain relevant information for this analisys
    list_filter.append('PRESCRIÇÃO FONOAUDIOLOGIA') # only one prescription items
    list_filter.append('PRESCRIÇÃO BIOMEDICO') # rare type of prescrition 
    list_filter.append('RECEITUÁRIO MÉDICO') # Prescription for generating document, does not contain relevant information

    list_nm_objeto = list(set(frame['NM_OBJETO']))
    for l in list_filter:
        list_nm_objeto.remove(l)
        frame = frame[frame.NM_OBJETO != l]

    return list_nm_objeto,frame

# =====================================================================

def get_cd_cid_list(frame,nm_objeto):

    '''
    Grup data in CD_CID, calculate the number of different attendances (QT_ATENDIMENTOS) and different number of prescriptions (QT_PRESCRICOES).
    Select the first ten diagnostics sorted by QT_PRESCRICOES
    '''


    group = frame.groupby(['CD_CID'],as_index=False).agg({"CD_ATENDIMENTO_INDEX":['nunique'],"CD_PRE_MED_INDEX":['nunique']})
    group.columns = ['CD_CID','QT_ATENDIMENTOS','QT_PRESCRICOES']

    groups = group.sort_values(by='QT_PRESCRICOES', ascending=False)
    groups['QT_ACUMULADA'] = groups.QT_PRESCRICOES.cumsum()
    groups['QT_ACUMULADA_PERC'] = 100 * groups.QT_ACUMULADA / groups.QT_PRESCRICOES.sum()

    writer_cd_cid_list = pd.ExcelWriter('.\\results\\' + nm_objeto + '_list_cd_cid.xlsx')
    groups.to_excel(writer_cd_cid_list,'Sheet1')
    writer_cd_cid_list.save()

    groups = groups.head(10)

    return list(set(groups["CD_CID"]))

# =====================================================================

def get_cd_sgru_cid_list(frame,nm_objeto):

    '''
    Grup data in CD_SGRU_CID, calculate the number of different attendances (QT_ATENDIMENTOS) and different number of prescriptions (QT_PRESCRICOES).
    Select the first ten diagnostics sorted by QT_PRESCRICOES
    '''

    group = frame.groupby(['CD_SGRU_CID'],as_index=False).agg({"CD_ATENDIMENTO_INDEX":['nunique'],"CD_PRE_MED_INDEX":['nunique']})
    group.columns = ['CD_CID','QT_ATENDIMENTOS','QT_PRESCRICOES']

    groups = group.sort_values(by='QT_PRESCRICOES', ascending=False)
    groups['QT_ACUMULADA'] = groups.QT_PRESCRICOES.cumsum()
    groups['QT_ACUMULADA_PERC'] = 100 * groups.QT_ACUMULADA / groups.QT_PRESCRICOES.sum()

    writer_cd_cid_list = pd.ExcelWriter('.\\results\\' + nm_objeto + '_list_cd_sgru_cid.xlsx')
    groups.to_excel(writer_cd_cid_list,'Sheet1')
    writer_cd_cid_list.save()

    groups = groups.head(10)

    return list(set(groups["CD_SGRU_CID"]))

# =====================================================================
def group_pre_med(frame,cd_cid,nm_objeto):

    '''
    Transform the data set to algorithms.
    Frist, pivot_table by DS_TIP_PRESC
    Secondly, release the non-pertinetes columns.
    Then, the transactions grouped by CD_PRE_MED_INDEX
    Finally transpose the matrix and calculate the median of the amount of prescription to generate the number of clusters
    '''

    frame = frame.drop(['DS_CID','CD_TIP_PRESC_INDEX'], axis=1)

    my_cols = set(frame.columns)
    my_cols.remove('DS_TIP_PRESC')
    my_cols = list(my_cols)

    table = pd.pivot_table(frame, values='DS_TIP_PRESC', index= my_cols,
                        columns=['DS_TIP_PRESC'], aggfunc=len, fill_value=0)
    
    writer_pivot = pd.ExcelWriter('.\\results\\' + nm_objeto + '_' + cd_cid + '_pivot_table.xlsx')
    table.to_excel(writer_pivot,'Sheet1')
    writer_pivot.save()

    frame = table.reset_index(my_cols)

    frame = frame.drop(['NM_OBJETO','NR_DIAS_PRESCIACAO','CD_PACIENTE_INDEX','DS_TIP_ESQ','CD_ATENDIMENTO_INDEX','NR_DIAS_NO_ATENDIMENTO','TP_SEXO','CD_TIP_ESQ','TP_ATENDIMENTO','CD_CID','DS_SGRU_CID','CD_SGRU_CID'], axis=1)

    my_cols = set(frame.columns)
    my_cols.remove('CD_PRE_MED_INDEX')
    my_cols = list(my_cols)

    group = frame.groupby(['CD_PRE_MED_INDEX'],as_index=True)[my_cols].sum()

    writer_pre_med = pd.ExcelWriter('.\\results\\' + nm_objeto + '_' + cd_cid + '_group_data.xlsx')
    group.to_excel(writer_pre_med,'Sheet1')
    writer_pre_med.save()
    
    grup_sum = group.sum(axis=1)

    median = np.median(grup_sum)

    group_T = group.T
    writer_pre_med_t = pd.ExcelWriter('.\\results\\' + nm_objeto + '_' + cd_cid + '_group_data_t.xlsx')
    group_T.to_excel(writer_pre_med_t,'Sheet1')
    writer_pre_med_t.save()

    return group_T,median

# =====================================================================

def get_features(frame):
    '''
    Transforms and scales the input data and returns a numpy array that
    is suitable for use with scikit-learn.

    Note that in unsupervised learning there are no labels.
    '''

    # Convert values to floats
    arr = np.array(frame, dtype=np.float)

    # Impute missing values from the mean of their entire column
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy='mean')
    arr = imputer.fit_transform(arr)
    
    # Normalize the entire data set to mean=0.0 and variance=1.0
    from sklearn.preprocessing import scale
    arr = scale(arr)

    return arr


# =====================================================================
def reduce_dimensions(X):
    '''
    Reduce the dimensionality of X with different reducers.

    Return a sequence of tuples containing:
        (title, x coordinates, y coordinates)
    for each reducer.
    '''

    # Principal Component Analysis (PCA) is a linear reduction model
    # that identifies the components of the data with the largest
    # variance.
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    X_r = reducer.fit_transform(X)
    yield 'PCA', X_r[:, 0], X_r[:, 1]

    # Independent Component Analysis (ICA) decomposes a signal by
    # identifying the independent contributing sources.
    from sklearn.decomposition import FastICA
    reducer = FastICA(n_components=2)
    X_r = reducer.fit_transform(X)
    yield 'ICA', X_r[:, 0], X_r[:, 1]

    # t-distributed Stochastic Neighbor Embedding (t-SNE) is a
    # non-linear reduction model.  It operates best on data with a low
    # number of attributes (<50) and is often preceded by a linear
    # reduction model such as PCA.
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2)
    X_r = reducer.fit_transform(X)
    yield 't-SNE', X_r[:, 0], X_r[:, 1]

# =====================================================================
def evaluate_learners(X,pre_med_mean):
    '''
    Run multiple times with different learners to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, predicted classes)
    for each learner.
    '''

    from sklearn.cluster import (MeanShift, MiniBatchKMeans,
                                 SpectralClustering, AgglomerativeClustering,Birch,DBSCAN)

    print('Evaluate - Mean Shift clusters')
    learner = MeanShift(# Let the learner use its own heuristic for determining the
        # number of clusters to create
        bandwidth=None)
    y = learner.fit_predict(X)
    yield 'Mean Shift clusters', y

    n_clustrs = int(round(X.shape[0] / max(pre_med_mean,10) - 0.5))

    print('Evaluate - K Means clusters (N={})'.format(n_clustrs))
    learner = MiniBatchKMeans(n_clusters=n_clustrs)
    y = learner.fit_predict(X)
    yield 'K Means clusters (N={})'.format(n_clustrs), y

    print('Evaluate - Spectral clusters (N={})'.format(n_clustrs))
    learner = SpectralClustering(n_clusters=n_clustrs)
    y = learner.fit_predict(X)
    yield 'Spectral clusters (N={})'.format(n_clustrs), y

    print('Evaluate - Agglomerative clusters (N={})'.format(n_clustrs))
    learner = AgglomerativeClustering(n_clusters=n_clustrs)
    y = learner.fit_predict(X)
    yield 'Agglomerative clusters (N={})'.format(n_clustrs), y

    print('Evaluate - Birch (N={})'.format(n_clustrs))
    learner = Birch(n_clusters=n_clustrs)
    y = learner.fit_predict(X)
    yield 'Birch  (N={})'.format(n_clustrs), y

    print('Evaluate - DBSCAN')
    learner = DBSCAN()
    y = learner.fit_predict(X)
    yield 'DBSCAN', y

# =====================================================================
def plot(Xs, predictions,cd_cid,nm_objeto):
    '''
    Create a plot comparing multiple learners.

    `Xs` is a list of tuples containing:
        (title, x coord, y coord)
    
    `predictions` is a list of tuples containing
        (title, predicted classes)

    All the elements will be plotted against each other in a
    two-dimensional grid.
    '''

    # We will use subplots to display the results in a grid
    nrows = len(Xs)
    ncols = len(predictions)

    fig = plt.figure(figsize=(16, 8))
    fig.canvas.set_window_title('Clustering data from ' + File)

    # Show each element in the plots returned from plt.subplots()
    
    for row, (row_label, X_x, X_y) in enumerate(Xs):
        for col, (col_label, y_pred) in enumerate(predictions):
            ax = plt.subplot(nrows, ncols, row * ncols + col + 1)
            if row == 0:
                plt.title(col_label)
            if col == 0:
                plt.ylabel(row_label)

            # Plot the decomposed input data and use the predicted
            # cluster index as the value in a color map.
            plt.scatter(X_x, X_y, c=y_pred.astype(np.float), cmap='prism', alpha=0.5)
            
            # Set the axis tick formatter to reduce the number of ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Let matplotlib handle the subplot layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    # plt.show()

    # To save the plot to an image file, use savefig()
    plt.savefig('.\\results\\' + nm_objeto + '_' + cd_cid + '_plot.png')

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================

def assing_data(frame, predictions,cd_cid,nm_objeto):
    '''
    Assign Data to Clusters
    '''

    my_cols = set(frame.columns)
    my_cols = list(my_cols)

    for predicion in predictions:
        frame[predicion[0]] = predicion[1]

    frame = frame.drop(my_cols, axis=1)

    writer_frame = pd.ExcelWriter('.\\results\\' + nm_objeto + '_' + cd_cid + '_assing_data.xlsx')
    frame.to_excel(writer_frame,'Sheet1')
    writer_frame.save()

# =====================================================================

if __name__ == '__main__':
    print("Clear old Results")
    clear_old()

    # Read dataset from File
    print("Reading data from {}".format(File))
    #dataset = read_xlsx_data()
    dataset = read_csv_data()

    # Get list for filtred NM_OJETO
    print("Filter invalid objects")
    list_nm_objeto,dataset = get_nm_objeto_list(dataset)

    for nm_objeto in list_nm_objeto:
        # Filter dataset by NM_OBJETO
        print("Filter dataset by objeto = {}".format(nm_objeto))
        frame_objeto = dataset.loc[dataset['NM_OBJETO'] == nm_objeto]

        # Get list for the most common CIDs
        print("Generate cid list by objeto = {}".format(nm_objeto))
        list_cd_cid = get_cd_cid_list(frame_objeto,nm_objeto)

        for cd_cid in list_cd_cid:

            # Group data by pre_med
            print("Transform data")
            frame_pre_med,pre_med_mean = group_pre_med(frame_objeto.loc[frame_objeto['CD_CID'] == cd_cid],cd_cid,nm_objeto)

            if(len(frame_pre_med.columns) > 10 and len(frame_pre_med.index) > 10):

                # Process data into a feature array
                # This is unsupervised learning, and so there are no labels
                print("Processing {} samples with {} attributes for {} objeto {}".format(len(frame_pre_med.index), len(frame_pre_med.columns),cd_cid,nm_objeto))
                X = get_features(frame_pre_med)

                # Run multiple dimensionality reduction algorithms on the data
                print("Reducing dimensionality")
                Xs = list(reduce_dimensions(X))

                # Evaluate multiple clustering learners on the data
                print("Evaluating clustering learners")
                predictions = list(evaluate_learners(X,pre_med_mean))

                # Display the results
                print("Plotting the results")
                plot(Xs, predictions,cd_cid,nm_objeto)

                # Display the results
                print("Assign Data to Clusters")
                assing_data(frame_pre_med, predictions,cd_cid,nm_objeto)
