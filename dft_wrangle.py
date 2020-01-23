# -*- coding: utf-8 -*-
"""
DFT hack data wrangling
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def get_slice(slice_path, month):
    slice_file = os.path.join(slice_path, 'lte_{}_18.csv'.format(month))
    df = pd.read_csv(slice_file, sep='\t', header=(0))
    return df


def agg_months(month_dict, months):
    #Accumulate all the data into a single frame
    df = month_dict['Jun'].copy()
    for month in months[1:]:
        df = df.append(month_dict[month], ignore_index=True)
    return df


def bin_lat_long(df, num_bins):
    df_bin = df.assign(
            lat_cut=pd.qcut(df['latitude'], num_bins, labels=range(num_bins)),
            long_cut=pd.qcut(df['longitude'], num_bins, labels=range(num_bins))
    )
    return df_bin


def smooth_feature(df_bin, feature):
    df_smooth = df_bin.groupby(['lat_cut', 'long_cut']).agg({feature:'mean'})
    df_smooth.reset_index(inplace=True)
    df_smooth.rename({feature:'mean'}, axis=1, inplace=True)
    df_bin = df_bin.merge(df_smooth, how='left', on=['lat_cut', 'long_cut'])
    return df_bin


if __name__ == "__main__":
    

    # %% Get 4g data
    path_data = 'C:\\Users\\Alex White\\Desktop\\dft_hack_data'
    path_yellow = os.path.join(path_data, 'yellow_slice', 'yellowTrain_lte(4g)_2018-19_slice')
    month_agg_4g = os.path.join(path_yellow, '4g_monthly_agg.csv')
    
    if not os.path.exists(os.path.join(month_agg_4g)):
        #Acquire monthly files
        month_dict = {}
        months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
        for month in months:
            df_month = get_slice(path_yellow, month)
            month_dict[month] = df_month
            
        #Fuse into a single file
        df_agg = agg_months(month_dict, months)
        df_agg.to_csv(month_agg_4g, index=False)
    else:
        df_agg = pd.read_csv(month_agg_4g)
        

            
    # %% Visualisation
    plt.close('all')
    pow_low  = df_agg['total_power'].quantile(0.25)
    pow_high = df_agg['total_power'].quantile(0.75)
    num_bins = 50
    extent_lat  = (df_agg['latitude'].min(), df_agg['latitude'].max())
    extent_long = (df_agg['longitude'].min(), df_agg['longitude'].max())
    
    #Bin spatially
    df_agg = bin_lat_long(df_agg, num_bins)
    print('LAT bin width: {0:.2f} degrees'.format((extent_lat[1] - extent_lat[0])/num_bins))
    print('LONG bin width: {0:.2f} degrees'.format((extent_long[1] - extent_long[0])/num_bins))
    
    fig, ax = plt.subplots()
    
    operators = dict(list(df_agg.groupby('operator')))
    op_colors = {'EE':'dodgerblue', 'O2':'crimson', 'Three':'green', 'Vodafone':'gold'}
    
    for ops in operators.keys():
        
        df_ops = operators[ops]

        #Distribution of signal power for the Leeds/liverpool slice
        power_data = df_ops['total_power'].dropna()
        
        kde = stats.gaussian_kde(power_data.values)
        xx = np.linspace(power_data.min(), power_data.max(), 10)
        ax.plot(xx, kde(xx), c=op_colors[ops])
#        ax.hist(df_ops['total_power'].dropna().values, density=True, bins=30, facecolor=op_colors[ops], edgecolor='k', alpha=0.2)
        ax.set_xlabel('Total signal power')
        ax.set_ylabel('Frequency density')
        
        #Smooth within each spatail bin
        df_ops = smooth_feature(df_ops, 'total_power')
        
        #Spatial location of all the measurements - according to 'low', 'medium', 'high'
        cond_high = (df_ops['mean'] > pow_high)
        cond_low  = (df_ops['mean'] < pow_low)
        df_low  = df_ops.loc[cond_low] 
        df_high = df_ops.loc[cond_high]
        df_med  = df_ops.loc[~cond_high & ~cond_low]
        
        fig_loc, ax_loc = plt.subplots()
        ax_loc.scatter(df_low['longitude'].values, df_low['latitude'].values, facecolor='crimson', alpha=0.1, s=25)
        ax_loc.scatter(df_med['longitude'].values, df_med['latitude'].values, facecolor='gold', alpha=0.1, s=25)
        ax_loc.scatter(df_high['longitude'].values, df_high['latitude'].values, facecolor='green', alpha=0.1, s=25)
        ax_loc.set_xlabel('Longitude / degrees')
        ax_loc.set_ylabel('Latitude / degrees')
        ax_loc.set_title('{}'.format(ops))
#        fig_loc.set_size_inches(16,12)
        plt.savefig(os.path.join(path_yellow, 'figures', '{}.png'.format(ops)), dpi=100)
    
    ax.set_title('4g signal strength between Liverpool and Leeds')
    ax.legend(['EE', 'O2', 'Three', 'Vodafone'])

#kimberley.brett@dft.gov.uk