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


#####################
##  Get GAIA data  ##
#####################
def get_GAIA_data(name, radius = 1, table = 'gaiadr3.gaia_source', 
                  search = True, preprocess = True, *args, **kwargs):

    if search:
        # conesearch using the object name and radius
        coordinate = coord.SkyCoord.from_name(name)
        print(f'Object Name: {name}')
        print('Co-ordinate', coordinate)

        radius = u.Quantity(radius, u.deg)
        Gaia.ROW_LIMIT = -1
        j = Gaia.cone_search_async(coordinate, radius, table_name = table)
        r = j.get_results()

        all_stars = r.to_pandas()
        print(f'Total Raw Stars: {len(all_stars)}')

    ## plotting the skyplot 
    sns.set(rc={'figure.figsize':(8.7,6.27)})
    skyplot = sns.scatterplot(x='ra', y='dec', data = all_stars)
    skyplot.invert_xaxis()
    plt.title('Skyplot of GAIA data')
    plt.show()

    if preprocess:
        ## magnitude error
        ## del magnitude = - 2.5 log(del Flux / Flux)
        all_stars['g_mag_error'] = 2.5/np.log(10) / all_stars.phot_g_mean_flux_over_error

        mask = (all_stars.parallax_over_error >= 3) & (all_stars.pmra_error < 1) \
                & (all_stars.pmdec_error < 1)

        all_stars = all_stars.loc[mask, :]
        print(f'Number of stars after applying noise filter: {len(all_stars)}')

        # calculating the distance from the parallax
        all_stars['distance_pc'] = 1/(all_stars.parallax*0.001)

        # positive parallax
        all_stars = all_stars[all_stars['parallax'] >= 0]

        # dropping rows with null values in required columns
        all_stars = all_stars[all_stars.loc[:, 'pmra'].notnull()]
        all_stars = all_stars[all_stars.loc[:, 'parallax'].notnull()]
        all_stars = all_stars[all_stars.loc[:, 'bp_rp'].notnull()]

        # defining proper motion (pm) range
        all_stars = all_stars[(abs(all_stars['pmra']) < 20) & (abs(all_stars['pmdec']) < 20)]

        # taking stars within 30' = 0.5 deg radius
        # all_stars = all_stars[all_stars['dist'] < (30/60)]
        print(f'Number of stars after applying other filters: {len(all_stars)}')

    return all_stars


#####################
## Get Cantat Data ##
#####################
def get_cantat_data(data = 'member', clusters = None):
    if data == 'cluster':
        if clusters == None:
            cantat_data = Vizier(catalog = 'J/A+A/633/A99/table1', 
                             row_limit = -1).query_constraints()
        else:
            cantat_data = Vizier(catalog = 'J/A+A/633/A99/table1', 
                                row_limit = -1).query_constraints(Cluster = clusters)
        cantat_data = cantat_data[0].to_pandas()
        cantat_data['Cluster'] = cantat_data.Cluster.apply(lambda x: x.decode('utf-8'))
        return cantat_data

    if data == 'member':
        if clusters == None:
            cantat_data = Vizier(catalog = 'J/A+A/633/A99/members', 
                             row_limit = -1).query_constraints()
        else:
            cantat_data = Vizier(catalog = 'J/A+A/633/A99/members', 
                                row_limit = -1).query_constraints(Cluster = clusters)
        cantat_data = cantat_data[0].to_pandas()
        cantat_data['Cluster'] = cantat_data.Cluster.apply(lambda x: x.decode('utf-8'))

        # renaming the cantat table to match it with gaia_data
        cantat_data = cantat_data.rename(columns={'Source':'source_id',
                                                'Proba':'PMemb',
                                                'RA_ICRS': 'ra',
                                                'DE_ICRS': 'dec',
                                                'pmRA': 'pmra',
                                                'pmDE': 'pmdec',
                                                'Gmag': 'phot_g_mean_mag',
                                                'BP-RP': 'bp_rp',
                                                'Plx': 'parallax'})
        
        return cantat_data

    

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


########################
##  Cantat Comparison ##
########################
def compare_with_cantat(cluster_name, cantat_member, gmm_member,
                        alpha = 0.2, selected_columns = ['ra', 'dec', 'pmra', 'pmdec',
                                                                       'parallax', 'bp_rp', 'phot_g_mean_mag','PMemb']):
    
    cantat_member = cantat_member[selected_columns]
    gmm_member = gmm_member[selected_columns]

    concatenated = pd.concat([gmm_member.assign(dataset='member_by_GMM'), 
                            cantat_member.assign(dataset='cantat')])
    # concatenated.reset_index(drop=True)
    print(f'Cluster: {cluster_name}')
    print(concatenated.dataset.value_counts())
    print(f'Cantat (PMemb > 0.5): {sum(cantat_member.PMemb > 0.5)}')
    print('\n')
    
    # Distribution of parameters
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    fig.suptitle(f"Distribution of the Cantat and predicted Members of {cluster_name.replace('_', ' ')}")

    sns.histplot(data = concatenated.reset_index(), x = 'parallax', element='step',
                 hue='dataset', ax=axes[0], palette = 'Dark2', alpha = alpha)
    axes[0].set_title('Parallax Distribution')
    
    sns.histplot(data = concatenated.reset_index(), x = 'pmra', element='step',
                 hue='dataset', ax=axes[1], palette = 'Dark2', alpha = alpha)
    axes[1].set_title('pmra Distribution')

    sns.histplot(data = concatenated.reset_index(), x = 'pmdec', element='step',
                 hue='dataset', ax=axes[2], palette = 'Dark2', alpha = alpha)
    axes[2].set_title('pmdec Distribution')
    plt.show()
                 
    # comparison of skyplot, pmplot and CMD
    fig, axes = plt.subplots(1, 3, figsize=(22,6))
    fig.suptitle(f"Predicted Members and Cantat Members of {cluster_name.replace('_', ' ')}")

    skyplot = sns.scatterplot(x='ra', y='dec', data=concatenated,
                    hue='dataset', ax=axes[0], palette = 'Dark2',
                    style='dataset')
    axes[0].set_title('Sky plot')

    # proper motion plot
    sns.scatterplot(x='pmra', y='pmdec', data=concatenated,
                    hue='dataset', ax=axes[1], palette = 'Dark2',
                    style='dataset')
    axes[1].set_title('Proper motion plot')

    cmd = sns.scatterplot(x='bp_rp', y='phot_g_mean_mag', data=concatenated,
                    hue='dataset', ax=axes[2], palette = 'Dark2',
                    style='dataset')
    cmd.invert_yaxis()
    axes[2].set_title('CMD')

    plt.show()