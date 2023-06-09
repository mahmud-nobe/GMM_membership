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
    
##################
##  Get RF Data ##
##################
def get_rf_member(cluster_name):
    rf_data = pd.read_csv('https://raw.githubusercontent.com/mahmud-nobe/Cluster-Membership/master/all_possible_members.csv', index_col=0)
    rf_member = rf_data[rf_data.cluster == cluster_name]

    if len(rf_member) == 0:
        print(f'The cluster, {cluster_name}, is not present in RF data.\n')
    return rf_member    

########################
##  Cantat Comparison ##
########################
def line_change(ax, linestyles):
    '''Add different linestyles for each group'''
    handles = ax.legend_.legendHandles[::-1]

    for line, ls, handle in zip(ax.collections, linestyles, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)
        
def compare_with_cantat(cluster_name, cantat_member, dbscan_member, rf_member,
                        alpha = 0.2, selected_columns = ['ra', 'dec', 'pmra', 'pmdec',
                                                                       'parallax', 'bp_rp', 'phot_g_mean_mag','PMemb']):
    
    cantat_member = cantat_member[selected_columns]
    dbscan_member = dbscan_member[selected_columns]
    rf_member = rf_member[selected_columns]

    concatenated = pd.concat([dbscan_member.assign(dataset='DBSCAN'), 
                            rf_member.assign(dataset = 'RF'),
                            cantat_member.assign(dataset='Cantat')])
    # concatenated.reset_index(drop=True)
    print(f'Cluster: {cluster_name}')
    print(concatenated.dataset.value_counts())
    print(f'Cantat (PMemb > 0.5): {sum(cantat_member.PMemb > 0.5)}')
    print('\n')

    # Distribution of parameters
    lss = [':','--','-']
    colors = {'DBSCAN' : '#1B9E77',
              'RF'  :  '#7570B3',
              'Cantat' : '#D95F02'}
    
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    fig.suptitle(f"Distribution of the Cantat and predicted Members of {cluster_name.replace('_', ' ')}")


    sns.histplot(data = concatenated.reset_index(), x = 'parallax', element='step',
                 hue='dataset', ax=axes[0], palette = colors, alpha = alpha)
    axes[0].set_title('Parallax Distribution')
    axes[0].set_xlabel('parallax (mas)')
    axes[0].get_legend().set_title(None)
    line_change(axes[0], lss)

    sns.histplot(data = concatenated.reset_index(), x = 'pmra', element='step',
                 hue='dataset', ax=axes[1], palette = colors, alpha = alpha)
    axes[1].set_title('pmra Distribution')
    axes[1].set_xlabel('pmra (mas/yr)')
    sns.move_legend(axes[1], "upper left")
    axes[1].get_legend().set_title(None)
    line_change(axes[1], lss)


    sns.histplot(data = concatenated.reset_index(), x = 'pmdec', element='step',
                 hue='dataset', ax=axes[2], palette = colors, alpha = alpha)
    axes[2].set_title('pmdec Distribution')
    axes[2].set_xlabel('pmdec (mas/yr)')
    sns.move_legend(axes[2], "upper left")
    axes[2].get_legend().set_title(None)
    line_change(axes[2], lss)

    plt.show()

    # comparison of skyplot, pmplot and CMD
    marker_dict = {'DBSCAN' : '^',
                   'RF': 'X',
                   'Cantat' : 'P'}

    fig, axes = plt.subplots(1, 3, figsize=(22,6))
    fig.suptitle(f"Predicted Members and Cantat Members of {cluster_name.replace('_', ' ')}")

    skyplot = sns.scatterplot(x='ra', y='dec', data=concatenated,
                            hue='dataset', ax=axes[0], palette = colors,
                            style='dataset',s=s, markers = marker_dict)
    axes[0].set_title('Sky plot')
    axes[0].set_xlabel('RA (deg)')
    axes[0].set_ylabel('Dec (deg)')
    axes[0].get_legend().set_title(None)

    # proper motion plot
    sns.scatterplot(x='pmra', y='pmdec', data=concatenated,
                    hue='dataset', ax=axes[1], palette = colors,
                    style='dataset', s=s, markers = marker_dict)
    axes[1].set_title('Proper motion plot')
    axes[1].set_xlabel('pmra (mas/yr)')
    axes[1].set_ylabel('pmdec (mas/yr)')
    axes[1].get_legend().set_title(None)

    cmd = sns.scatterplot(x='bp_rp', y='phot_g_mean_mag', data=concatenated,
                    hue='dataset', ax=axes[2], palette = colors,
                    style='dataset', s=s, markers = marker_dict)
    cmd.invert_yaxis()
    axes[2].set_title('CMD')
    axes[2].set_xlabel('G Magnitude')
    axes[2].set_ylabel('G - RP Color')
    axes[2].get_legend().set_title(None)

    plt.show()