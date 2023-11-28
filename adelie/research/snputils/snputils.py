import adelie.logger as logger
import pandas as pd
import numpy as np
from tqdm import tqdm


def phe_to_csv(
    src: str,
    phenotype: str,
    dest: str,
):
    master_df = pd.read_csv(src, sep='\t', usecols=['IID', phenotype])
    master_df.to_csv(dest, sep=",", index=False)
    return master_df


def pgen_to_csv(
    src: str,    
    dest: str,
):
    """Converts a PGEN file to CSV file.

    Parameters
    ----------
    src : str
        PGEN filename.
    dest : str
        CSV filename.
    """
    # pgenlib is a really poorly written library that doesn't work on Mac M1.
    # We do not perform a global import since this module cannot be loaded on Mac M1 then.
    import pgenlib as pg

    encoded_pgen_path = str.encode(src)
    pgen_reader = pg.PgenReader(
        encoded_pgen_path,
    )

    calldata_shape = (pgen_reader.get_variant_ct(), 2 * pgen_reader.get_raw_sample_ct())
    logger.logger.info(f"calldata shape: {calldata.shape}")
    calldata = np.empty(
        calldata_shape,
        dtype=np.int8,
    )

    logger.logger.info(f"Reading PGEN file.")
    pgen_reader.read_alleles_range(0, pgen_reader.get_variant_ct(), calldata)
    pgen_reader.close() 

    logger.logger.info(f"Saving calldata as CSV.")
    np.savetxt(dest, calldata, delimiter=",")

    return calldata


def msp_to_csv(
    src: str,
    dest: str,
):
    """Converts MSP file to CSV.

    Parameters
    ----------
    src : str
        MSP filename.
    dest : str
        CSV filename.
    """
    import msp_reader
    reader = msp_reader.MSPReader(src)

    logger.logger.info(f"Reading MSP file.")
    snpobj = reader.read()

    logger.logger.info(f"Saving LAI as CSV.")
    np.savetxt(dest, snpobj.lai, delimiter=",")

    return snpobj.lai


def common_iids(
    phe: str,
    msp: str,
    psam: str, 
):
    # get non-missing IIDs in phenotype
    master_df = pd.read_csv(phe)
    phe_iids = (master_df.iloc[:, 1] != -9).to_numpy()
    phe_iids = master_df.iloc[phe_iids, 0]

    # get MSP IIDs and intersect with previous
    import msp_reader
    reader = msp_reader.MSPReader(msp)
    header = reader.read_header()
    msp_iids = reader.get_iids(header)
    msp_iids = pd.Series([int(x) for x in msp_iids])
    msp_iids = msp_iids[msp_iids.isin(phe_iids)]

    # get calldata IIDs and intersect with previous
    samples_info = pd.read_csv(psam, sep='\t')
    psam_iids = samples_info["IID"]
    psam_iids = samples_info.loc[psam_iids.isin(msp_iids).to_numpy(), 'IID']

    common_iids = psam_iids

    return (
        phe_iids[phe_iids.isin(common_iids)].index.to_numpy(dtype=np.uint32), 
        msp_iids[msp_iids.isin(common_iids)].index.to_numpy(dtype=np.uint32), 
        psam_iids.index.to_numpy(dtype=np.uint32),
        psam_iids.to_numpy(dtype=np.uint32),
    )