# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:35:55 2017

@author: Raphael Mikati

Description : This script performs the fitting process.
    > search_sup is an auxiliary function aimed at finding the Sup of two function 
    > fitting_algo is aimed at finding the statistical distribution that would
    best fit the empirical one """

# Importing relevant packages
import numpy 
import pandas 
import statsmodels as sm
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.multitest import multipletests

# Auxiliary function aimed at computing a Sup between two 
def search_sup (tab1, tab2):
    """Function aimed at finding the Sup of the difference of two distributions 
    
    Parameters
    -----------
    tab1 : a list or an array containing data
    tab2 : a list or an array containing data
    
    Returns
    -----------
    Sup : a float representing Sup(abs(tab1[i] - tab2[i]) """
    len_tab = len(tab1)
    assert len_tab == len(tab2)
    Sup = -1.
    for k in range(len_tab):
        diff = abs(tab1[k]-tab2[k])
        if diff > Sup :
            Sup = diff
    return(Sup)
    
# Creating a pool of candidates laws    
#DISTRIBUTIONS = [st.expon, st.lognorm, st.pareto, st.powerlaw, st.norm]#, st.beta, st.gamma, st.rayleigh, st.weibull_max, st.weibull_min]       
    
# Function aimed at fitting the empirical data distribution
def fitting_algo(data, DISTRIBUTIONS):
    """Function aimed at a given data distribution to a statistical laws 
    
    Parameters
    -----------
    data : an array containing data 
    DISTRIBUTIONS : a list of statistical distributions considered candidates
    
    Returns
    -----------
    display : a Pandas series giving the name and the parameters of the best
    candidate to fit the given data distribution """    
    nbr_candidates = len(DISTRIBUTIONS)
 
    # Initialisation of decision parameters
    error_min = numpy.inf
    best_distribution = "None"
    best_parameters = []
    pvalue = -1.
    
    #Initialising arrays
    tab_pvalues = []
    tab_laws = []
    tab_params = []
    tab_errors = []
    
    # Estimating the empirical cumulative distribution function
    data_sorted = numpy.sort(numpy.unique(data))
    ecdf = ECDF(data_sorted)
    array_ecdf = ecdf(data_sorted)
    
    # Assesing each candidate
    for distribution in DISTRIBUTIONS :
        tab_laws.append(distribution.name)
        # Determining the parameters that would best fit the dataset for the given law
        params = distribution.fit(data)
        tab_params.append(params)
        #Separating the parameters
        arg = params[:-2]
        loc = params[-2] 
        scale = params[-1] 
        
        # Extracting the candidate's CDF 
        cdf = distribution.cdf(data_sorted,loc=loc, scale=scale, *arg)
        
        # Implementing the Kolmogorov-Smirnov test
        # Computing Sup(abs(ecdf - cdf)) via an auxiliary function
        error = search_sup(array_ecdf, cdf)
        tab_errors.append(error)
        
        # Computing the p-value between the two cumulative distributions 
        pvalue_calc = st.ks_2samp(cdf, array_ecdf)[1]
        tab_pvalues.append(pvalue_calc)
        
       
    # Adjusting p-value regarding the multiple hypothesis testing risk (alpha = 5%), method : Holm-Bonferroni  
    adjustments = multipletests(tab_pvalues, alpha=0.05, method='h',is_sorted=False, returnsorted=False)
    tests_to_be_rejected = adjustments[0]
    adjusted_pvalues = adjustments[1]
    len_pvalues = len(adjusted_pvalues)
    
    # Checking whether this candidate is a better fit
    for i in range (len_pvalues):
        if (tests_to_be_rejected[i]==False and tab_errors[i]<error_min and adjusted_pvalues[i]>0.1):
            error_min = tab_errors[i]
            best_distribution = tab_laws[i]
            best_parameters = tab_params[i]
            pvalue = adjusted_pvalues[i]
    
    # Returning the result
    len_params = len(best_parameters) 
    if len_params == 2 :
        names = ["Law","Parameter_arg", "Parameter_loc", "Parameter_scale", "Kolmogorov_Smirnov_distance", "p_value"]
        l = [best_distribution, best_parameters[:-2],best_parameters[-2], best_parameters[-1], error_min, pvalue]
        display = pandas.Series(l, index = names)
        return(display)
    if len_params == 1 :
        names = ["Law","Parameter_arg", "Parameter_scale", "Kolmogorov_Smirnov_distance", "p_value"]
        l = [best_distribution, best_parameters[:-2], best_parameters[-1], error_min, pvalue]
        display = pandas.Series(l, index = names)
        return(display)
    else :
        names = ["Law","Parameter_arg", "Kolmogorov_Smirnov_distance", "p_value"]
        l = [best_distribution, best_parameters[:-2], error_min, pvalue]
        display = pandas.Series(l, index = names)
        return(display)
    
    

#data = retrieve_data_from_csv ("pareto.csv")#st.lognorm.rvs(2.05,size=13200)
#print(fitting_algo(data, DISTRIBUTIONS))