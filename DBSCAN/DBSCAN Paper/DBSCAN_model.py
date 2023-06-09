#####################
## import packages ##
#####################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

# import astroquery
import astropy.units as u
import astropy.coordinates as coord
from astroquery.gaia import Gaia
from astroquery.gaia import TapPlus, GaiaClass   
from astroquery.vizier import Vizier
import warnings
warnings.filterwarnings('ignore')


# import helper functions from data_collection.py
from data_collection import *
 
    
#################################
## Helper Functions for DBSCAN ##
#################################

def get_normalized_feature(all_stars, feature_columns = ['pmra', 'pmdec', 'parallax']):
    # selecting the features
    features = all_stars.loc[:, feature_columns]
    features = features.dropna()

    # normalizing the features
    scaled_features = pd.DataFrame({})
    for column in features.columns:
        scaled_features[column] = (features[column] - np.median(features[column]))/np.std(features[column])
    
    return scaled_features


def get_members(all_stars, eps, min_sample, feature_columns = ['pmra', 'pmdec', 'parallax']):
    
    scaled_features = get_normalized_feature(all_stars, feature_columns = feature_columns)
    
    db = DBSCAN(eps= eps,min_samples= min_sample).fit(scaled_features)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    all_stars['labels'] = labels
    all_stars['is_core'] = core_samples_mask

    member, non_member = all_stars[all_stars.labels == 0],\
                         all_stars[all_stars.labels == -1]
    return member, non_member


def mean_nearest_neighbor(member_data, feature_columns = ['pmra', 'pmdec', 'parallax']):
    '''
    returns the average of the nearest neighbor distance for all the member stars
    ''' 

    # normalizing the features 
    scaled_member_features = get_normalized_feature(member_data, feature_columns)

    nn_model = NearestNeighbors(n_neighbors=2) # model to find 1st nearest neighbor

    nn_members = nn_model.fit(scaled_member_features)        # training the model using normalized dataset 

    nn_distances, nn_indices = nn_members.kneighbors(scaled_member_features) #[(distance, index of the 30th nearest neighbor)]

    mnn = np.mean(nn_distances[:, 1])

    return mnn

def get_MSS_metric(member, non_member, epsilon = 1e-7, 
                   feature_columns = ['pmra', 'pmdec', 'parallax']):
    '''
    Returns the Modified Silhouttee Score for given member and non_member data
    '''
    if len(member) < 2 or len(non_member) < 2:
        return 0

    metric = np.zeros(len(feature_columns))
    for i in range(len(feature_columns)):
        feature_i = feature_columns[i]
        # (field_SD - member SD)/max(field_SD, member SD, epsilon) for each features
        metric[i] = (np.std(non_member[feature_i]) - np.std(member[feature_i])) \
                / max(np.std(member[feature_i]), np.std(non_member[feature_i]), epsilon)
    
    mss_metric = metric.mean()
    return mss_metric

####################################
## DBSCAN Choosing Hyperparameter ##
####################################

def compare_DBSCAN_parameters(all_stars, eps, min_samples, 
                              feature_columns = ['pmra', 'pmdec', 'parallax']):

    scaled_features = get_normalized_feature(all_stars, feature_columns = feature_columns)
    
    # running DBSCAN with all possible parameters
    mnn_values = np.full((len(eps), len(min_samples)), -1, dtype = 'float32')
    n_members = np.full((len(eps), len(min_samples)), -1, dtype = 'float32')
    mss_metrics = np.full((len(eps), len(min_samples)), -1, dtype = 'float32')

    for i in range(len(eps)):
        ep = eps[i]
        for j in range(len(min_samples)):
            min_sample = min_samples[j]

            member, non_member = get_members(all_stars, scaled_features, ep, min_sample)
            try:
                mnn_values[i,j] = mean_nearest_neighbor(member)
            # if there is any error comes up, we will continue the program by keeping -1 as the MNN value
            except: 
                pass
            mss_metrics[i,j] = get_MSS_metric(member, non_member)
            n_members[i,j] = len(member)

    # Creating a DataFrame with paramters and metrics
    model_parameters = pd.DataFrame({'eps': np.repeat(eps, len(min_samples)),
                        'min_sample': np.tile(min_samples, len(eps)),
                        'mnn': mnn_values.flatten(),
                        'mss': mss_metrics.flatten(), 
                        'n_member' : n_members.flatten()})
                      
    model_parameters = model_parameters[model_parameters.mnn > 0] # removing the rows where MNN is negative
    

    # Visualize the change in metric values
    plt.figure(figsize = (20,6))

    plt.subplot(131)
    sns.lineplot(x='eps', y='mnn', hue = 'min_sample', data = model_parameters, 
                 legend = 'full')

    plt.subplot(132)
    sns.lineplot(x='eps', y='mss', hue = 'min_sample', data = model_parameters, 
                 legend = 'full')

    plt.subplot(133)
    sns.lineplot(x='eps', y='n_member', hue = 'min_sample', data = model_parameters, 
                 legend = 'full')
    plt.show()

    return model_parameters

def save_members(cluster_name, members, output_file):
        members['Cluster'] = [cluster_name]*len(members)
        members.to_csv(output_file, index = False)

###########################
## Main Class: GMM Model ##
###########################
class Run_GMM_Model():
    '''
    This class runs GMM model to the selected part of the datset, predicts the member and field star
    group, calculates the evaluation metric and visualizes the results
    
    Attribute:
    ----------
    data:           Full dataset of stars (after applying noise filter)
    working_data:   Filtered dataset by a given half-width value. 
                    We will run GMM model in this filtered dataset
                    
    (The following attribute is used to determine the working_data)
    distance_lit:   Distance of the cluster from the literature 
    
    feature_colums: The list of columns used as the features of GMM model
    gmm:            The GMM model applied on working data
    member:         The predicted member stars by GMM model in working data
    non_member:     The predicted field stars by GMM model in working data
    mss_metric:     Modified Silhouttee Score (MSS) for the current member and non_member group
    '''
    
    def __init__(self, data, cluster_name, distance):
        '''
        Initializes the class with a dataset.
        
        Input: 
        -------
        data:         Full dataset of stars
        cluster_name: Name of the cluster
        distance:     Distance of the cluster from the literature 
        '''
        self.data = data
        self.distance_lit = distance
        self.cluster_name = cluster_name
        self.best_half_width = None
        self.best_member_cutoff = None
        
    def get_working_data(self, half_width):
        '''
        Returns the filtered subset of the datset based on the given half-width
        
        Input:
        ------
        half_width: The half-width for the distance cutoff
        '''
        filter = abs(self.data.distance_pc - self.distance_lit) < half_width
        working_data = self.data.loc[filter, :]
        return working_data
    
    def get_member(self, half_width, feature_columns, cutoff = 0.6, random_state = 42):
        '''
        Classifies the working data into member and non_member groups.
        
        Inputs:
        -------
        half_width:     The half-width for the distance cutoff
        feature_colums: The list of columns used as the features of GMM model
        cutoff:         Threshold used for member cutoff. Default is 0.6.
                        If cutoff is 0.6, stars with membership probability >= 0.6 are members
                        and the stars with membership probability <= 0.4 (1-0.6) are non-members
        random_state:   Random State of the GMM algorithm. Used for reproducibility.
                        If 'None', then each time a new random seed will be used. Default is None.
                        
        GMM clusters the data into two groups. The group which has smaller average standard deviation 
        (thus more compact) is defined as the member group.
        '''
        self.feature_columns = feature_columns
        self.working_data = self.get_working_data(half_width)
        
        features = self.working_data.loc[:,feature_columns].dropna()

        # if there is less than 2 stars, GMM cannot divide them in two groups
        if len(self.working_data) < 2:
            raise ValueError('Less than two stars in the data')

        # normalizing the features
        scaled_features = pd.DataFrame({})
        for column in features.columns:
            scaled_features[column] = (features[column] - np.median(features[column]))/np.std(features[column])

        # Running GMM model: n_init = Number of Different Initialization tried by GMM
        gmm = GaussianMixture(n_components=2, n_init = 5, random_state = random_state)
        gmm.fit(scaled_features)

        #predictions from gmm
        labels = gmm.predict(scaled_features)       # group no (0 or 1)
        probs = gmm.predict_proba(scaled_features)  # membership probability
        
        # first assuming the group 1 is the member group
        self.working_data.loc[:, 'PMemb'] = probs[:, 1]  # membership probability to be in group 1
        self.working_data.loc[:, 'gmm_label'] = labels

        # select member and non-member based on member threshold
        non_member_ind = self.working_data.loc[:, 'PMemb'] <= (1-cutoff)
        non_member = self.working_data.loc[non_member_ind, :]

        member_ind = self.working_data.loc[:, 'PMemb'] >= cutoff
        member = self.working_data.loc[member_ind, :]

        # check if the average standard deviation of member group is larger.
        # if yes, then change the member group and assign group 0 as the member group. 
        if member[feature_columns].std().mean() > non_member[feature_columns].std().mean():
            self.working_data.loc[:, 'PMemb'] = probs[:, 0] # membership probability to be in group 1
            self.working_data.loc[:, 'gmm_label'] = 1-labels

            # select member and non-member based on member threshold using new PMemb value
            non_member_ind = self.working_data.loc[:, 'PMemb'] <= (1-cutoff)
            non_member = self.working_data.loc[non_member_ind, :]

            member_ind = self.working_data.loc[:, 'PMemb'] >= cutoff
            member = self.working_data.loc[member_ind, :]

        # save the results as an attribute of the class
        self.member, self.non_member, self.gmm = member, non_member, gmm
    
    def get_MSS_metric(self, epsilon = 1e-7):
        '''
        Calculates modified silhouttee score (MSS) from the current member and non_member group.
        If any of the group has less than 2 member (thus SD = 0 or None), MSS value is 0.
        Otherwise, MSS is the average of (field_SD - member SD)/max(field_SD, member SD, epsilon) for all features.
        
        epsilon: a very small number to avoid dividing by 0, when both SD are 0. Default value 1e-7.
        '''
        if len(self.member) < 2 or len(self.non_member) < 2:
            return 0

        metric = np.zeros(len(self.feature_columns))
        for i in range(len(self.feature_columns)):
            feature_i = self.feature_columns[i]
            # (field_SD - member SD)/max(field_SD, member SD, epsilon) for each features
            metric[i] = (np.std(self.non_member[feature_i]) - np.std(self.member[feature_i])) \
                    / max(np.std(self.member[feature_i]), np.std(self.non_member[feature_i]), epsilon)
        
        self.mss_metric = metric.mean()
        return self.mss_metric
    
    def visualize_member(self, title = None):
        '''
        Visualizes the following four plots in a 2x2 (row x col) setup:
        1. Proper motion plot (pmra vs pmdec) for current member and non-member
        2. Parallax distribution for current member and non-member
        3. Color-Magnitude Diagram (CMD) for member group
        4. Color-Magnitude Diagram (CMD) for non-member group
        
        Input:
        -------
        title: An overall title of these set of figures
        '''
        plt.figure(figsize=(14,14))

        if title:
            plt.suptitle(title)

        plt.subplot(221)
        sns.scatterplot(data = self.non_member, x='pmra', y='pmdec', label = 'non_member', color = 'tab:blue')
        sns.scatterplot(data = self.member, x='pmra', y='pmdec', label = 'member', color = 'tab:red')
        plt.xlabel('pmra (mas/yr)')
        plt.ylabel('pmdec (mas/yr)')

        plt.subplot(222)
        sns.distplot(self.non_member.parallax, label = 'non_member', color = 'tab:blue', kde = True)
        sns.distplot(self.member.parallax, label = 'member', color = 'tab:red', kde = True)
        plt.xlabel('Parallax (mas)')
        plt.ylabel('Probability Density [$mas^{-1}$]')
        plt.legend()

        plt.subplot(223)
        sns.scatterplot(data = self.member, x='g_rp', y='phot_g_mean_mag', color = 'tab:red', label = 'member')
        plt.gca().invert_yaxis()
        plt.ylabel('G Magnitude')
        plt.xlabel('G - RP Color')

        plt.subplot(224)
        sns.scatterplot(data = self.non_member, x='g_rp', y='phot_g_mean_mag', color = 'tab:blue', label = 'non member')
        plt.gca().invert_yaxis()
        plt.ylabel('G Magnitude')
        plt.xlabel('G - RP Color')

        plt.show()
        
    
    def save_members(self, output_file):
        self.member['Cluster'] = [self.cluster_name]*len(self.member)
        self.member.to_csv(output_file, index = False)
        
        
#####################
##  Get GMM Member ##
#####################
def get_GMM_member(cluster_name, all_stars, cantat_clusters = get_cantat_data(data='cluster'),
                   member_cutoff = 0.9, visualize_member = True):
    
    # distance of the cluster taken from Cantat-Gaudin (2018)

    
    # range of half-widths
    half_widths = np.linspace(50, 1400, 28).round(2)
    half_widths

    distance_lit = cantat_clusters.loc[cantat_clusters.Cluster == cluster_name, 'dmode'].iloc[0]
    gmm_model = Run_GMM_Model(all_stars, cluster_name, distance_lit)

    metrics = np.full(len(half_widths), -1, dtype='float')
    n_members = np.full(len(half_widths), -1, dtype='float')

    n_trial = 15

    for i in range(len(half_widths)):
        metric_value = 0
        n_member = 0

        # we will take the average MSS over all the trials for a given set of half-width and n_field
        for _ in range(n_trial):
            half_width = half_widths[i]
            
            feature_columns = ['pmra', 'pmdec', 'parallax']
            # get member and non_member
            # we want randomness in every different trial, thus we kept random_state = None
            try:
                gmm_model.get_member(half_width, cutoff = member_cutoff, feature_columns = feature_columns,
                                    random_state = None)
            except ValueError as err:
                print(f'Half-width: {half_width}\n, Value Error: {err}')
                pass

            # number of star and metric calculation
            n_member += len(gmm_model.member)
            metric_value += gmm_model.get_MSS_metric()

        
        metrics[i] = metric_value/n_trial
        n_members[i] = n_member / n_trial
        
        if visualize_member:
            try:
                gmm_model.visualize_member(title=f'Half-width: {half_width:0.2f}')
            except BaseException as err:
                print(f"{type(err).__name__}: {err}")
                pass

    # Plotting the MSS vs distance cutoff
    plt.figure(figsize = (13, 6))

    plt.subplot(121)
    sns.lineplot(half_widths, metrics, label = 'Metric Value')
    plt.xlabel('Half-width of the distance cutoff')
    plt.ylabel('Modified silhouette score')
    plt.legend()

    plt.subplot(122)
    sns.lineplot(half_widths, n_members, label = 'n_Member')
    plt.xlabel('Half-width of the distance cutoff')
    plt.ylabel('Number of retrieved members')
    plt.legend()
    plt.show()

    # best half-width is defined when we get maximum MSS
    best_width = half_widths[np.nanargmax(metrics)]
    gmm_model.best_half_width = best_width
    print(f'best distance half-width: {best_width}')

    ## Member Threshold
    cutoffs = np.linspace(0.5, 0.95, 10)
    MSS_metrics = []
    n_members = []

    for cutoff in cutoffs:
        gmm_model.get_member(best_width, feature_columns, cutoff=cutoff)
        MSS = gmm_model.get_MSS_metric()
        
        MSS_metrics.append(MSS)
        n_members.append(len(gmm_model.member))

    # MSS vs member threshold plot
    plt.figure(figsize = (15, 6))
    sns.set(font_scale = 1.2)

    plt.subplot(121)
    sns.lineplot(cutoffs, MSS_metrics)
    plt.xlabel('Member cutoff')
    plt.ylabel('Modified silhouette score')

    plt.subplot(122)
    sns.lineplot(cutoffs, n_members)
    plt.xlabel('Member cutoff')
    plt.ylabel('Number of member retrieved')
    plt.show()

    best_cutoff = cutoffs[np.nanargmax(MSS_metrics)]
    gmm_model.best_member_cutoff = best_cutoff
    print(f'best member threshold: {best_cutoff}')

    # building the final GMM model
    cutoff = best_cutoff
    feature_columns = ['pmra', 'pmdec', 'parallax']

    gmm_model.get_member(best_width, feature_columns, best_cutoff, random_state = 42)
    MSS = gmm_model.get_MSS_metric()

    print(f'number of member: {len(gmm_model.member)},\n number of field star: {len(gmm_model.non_member)},\n MSS: {MSS:0.2f}') 
    gmm_model.visualize_member(title=f'half-width: {best_width}, cutoff: {cutoff}')

    return gmm_model

