import os, sys
import numpy as np
import pandas as pd
from lsst.utils import getPackageDir
from lsst.sims.photUtils import Sed, BandpassDict, Bandpass, CosmologyObject
from lsst.sims.utils import findHtmid
sys.path.insert(0, '../sims_GCRCatSimInterface/python/desc/sims/GCRCatSimInterface')
from AGNModule import M_i_from_L_Mass, log_Eddington_ratio, k_correction, tau_from_params, SF_from_params
import GCRCatalogs

def unravel_dictcol(agn_df):
    # Unravel the string into dictionary
    agn_df['varParamStr'] = agn_df['varParamStr'].apply(eval)
    # Convert the dictionary inside key 'p' of the 'varParamStr' dictionary into columns
    agn_params_df = (agn_df['varParamStr'].apply(pd.Series))['p'].apply(pd.Series)
    # Combine the agn parameters with the original df containing galaxy_id and magNorm
    agn_df = pd.concat([agn_df.drop(['varParamStr'], axis=1), agn_params_df], axis=1)
    return agn_df

def create_k_corr_grid(redshift):
    """
    Returns a grid of redshifts and K corrections on the
    LSST Simulations AGN SED that can be used for K correction
    interpolation.
    """
    bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
    bp_i = bp_dict['i']
    sed_dir = os.path.join(getPackageDir('sims_sed_library'),
                           'agnSED')
    sed_name = os.path.join(sed_dir, 'agn.spec.gz')
    if not os.path.exists(sed_name):
        raise RuntimeError('\n\n%s\n\nndoes not exist\n\n' % sed_name)
    base_sed = Sed()
    base_sed.readSED_flambda(sed_name)
    z_grid = np.arange(0.0, redshift.max(), 0.01)
    k_grid = np.zeros(len(z_grid),dtype=float)

    for i_z, zz in enumerate(z_grid):
        ss = Sed(flambda=base_sed.flambda, wavelen=base_sed.wavelen)
        ss.redshiftSED(zz, dimming=True)
        k = k_correction(ss, bp_i, zz)
        k_grid[i_z] = k

    return z_grid, k_grid


def get_m_i(abs_mag_i, redshift):
    """
    Take numpy arrays of absolute i-band magnitude and
    cosmological redshift.  Return a numpy array of
    observed i-band magnitudes
    """
    z_grid, k_grid = create_k_corr_grid(redshift)
    k_corr = np.interp(redshift, z_grid, k_grid)

    dc2_cosmo = CosmologyObject(H0=71.0, Om0=0.265)
    distance_modulus = dc2_cosmo.distanceModulus(redshift=redshift)
    obs_mag_i = abs_mag_i + distance_modulus + k_corr
    return obs_mag_i

def join_catalogs(agn_df, loaded_cosmodc2):
    agn_galaxy_ids = agn_df['galaxy_id'].values
    
    agn_df = unravel_dictcol(agn_df)
    
    quantities = ['galaxy_id',
              'blackHoleAccretionRate', 'blackHoleEddingtonRatio', 'blackHoleMass',
              'redshift',]
    quantities += ['mag_true_%s_sdss' %bp for bp in 'ugriz']

    galaxy_id_min = np.min(agn_galaxy_ids)
    galaxy_id_max = np.max(agn_galaxy_ids)

    filters = ['galaxy_id >= %d' %(galaxy_id_min),
               'galaxy_id <= %d' %(galaxy_id_max)]

    cosmodc2_obj = loaded_cosmodc2.get_quantities(quantities, filters=filters)
    cosmodc2_df = pd.DataFrame(cosmodc2_obj)
    
    joined = pd.merge(cosmodc2_df, agn_df, on='galaxy_id')
    print(joined.shape)
    print(joined.columns)
    return joined

def add_columns(joined):
    black_hole_mass = joined['blackHoleMass'].values
    edd_ratio = joined['blackHoleEddingtonRatio'].values
    redshift = joined['redshift'].values
    z_corr = 1.0/(1.0 + redshift)
    wavelength_norm = 4000.0 # Angstroms

    joined['M_i'] = M_i_from_L_Mass(np.log10(edd_ratio), np.log10(black_hole_mass))
    joined['rf_u'] = 3520.0*z_corr/wavelength_norm
    joined['rf_g'] = 4800.0*z_corr/wavelength_norm
    joined['rf_r'] = 6250.0*z_corr/wavelength_norm
    joined['rf_i'] = 7690.0*z_corr/wavelength_norm
    joined['rf_z'] = 9110.0*z_corr/wavelength_norm
    joined['m_i'] = get_m_i(joined['M_i'].values, redshift)
    return joined