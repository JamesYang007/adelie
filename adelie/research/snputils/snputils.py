import logging
import adelie as ad
import adelie.logger as logger
import pandas as pd
import numpy as np
import os


def phe_to_csv(
    src: str,
    phenotype: str,
    dest: str,
):
    logger.logger.info("Reading master CSV.")
    master_df = pd.read_csv(src, sep='\t', usecols=['IID', phenotype])
    logger.logger.info(f"Saving IID/{phenotype} CSV.")
    master_df.to_csv(dest, sep=",", index=False)
    return master_df


def gen_to_snpdat(
    pgen: str,    
    pvar: str,
    psam: str,
    msp: str,
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

    # instantiate PGEN reader
    pgen_reader = pg.PgenReader(
        str.encode(pgen),
    )

    # create calldata array
    calldata_shape = (pgen_reader.get_variant_ct(), 2 * pgen_reader.get_raw_sample_ct())
    logger.logger.info(f"calldata shape: {calldata.shape}")
    calldata = np.empty(
        calldata_shape,
        dtype=np.int8,
    )

    # store calldata
    logger.logger.info(f"Reading PGEN file.")
    pgen_reader.read_alleles_range(0, pgen_reader.get_variant_ct(), calldata)
    pgen_reader.close() 

    # load PGEN metadata
    logger.logger.info(f"Loading PGEN metadata.")
    pvar_df = pd.read_csv(
        pvar, 
        sep='\t', 
        comment='#',
        header=None, 
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT'],
        dtype={'CHROM': str},
    )
    psam_df = pd.read_csv(psam, sep='\t')

    # instantiate MSP reader
    import msp_reader
    reader = msp_reader.MSPReader(msp)

    logger.logger.info(f"Reading MSP file.")
    snpobj = reader.read()

    # check that IIDs match up.
    logger.logger.info(f"Checking that IIDs are exactly the same in MSP and PSAM.")
    psam_iids = psam_df["IID"].to_numpy()
    psam_iids_order = np.argsort(psam_iids)
    msp_iids = np.array([int(i) for i in snpobj.sample_IDs], dtype=np.uint32)
    msp_iids_order = np.argsort(msp_iids)

    assert np.equal(psam_iids[psam_iids_order], msp_iids[msp_iids_order])

    # check that LAI SNPs contain all PVAR SNPs
    assert np.equal(np.sort(pvar_df["POS"]), pvar_df["POS"])
    assert snpobj.physical_pos[0, 0] <= pvar_df["POS"][0]
    assert snpobj.physical_pos[-1, 1] > pvar_df["POS"][-1]
    # TODO: check that physical_pos is always increasing windows?

    calldata = np.transpose(
        calldata.reshape((calldata.shape[0], calldata.shape[1], 2)),
        (1, 0, 2),
    )[psam_iids_order]
    calldata = calldata.reshape((calldata.shape[0], -1))
    calldata = np.array(calldata, copy=False, order="C", dtype=np.int8)

    lai = np.repeat(
        snpobj.lai,
        snpobj.physical_pos[:, 1] - snpobj.physical_pos[:, 0],
        axis=0,
    )
    lai = np.transpose(
        lai.reshape((lai.shape[0], lai.shape[1], 2)),
        (1, 0, 2),
    )[msp_iids_order]
    lai = lai.reshape((lai.shape[0], -1))
    lai = np.array(lai, copy=False, order="C", dtype=np.int8)

    assert calldata.shape == lai.shape

    # convert to snpdat
    handler = ad.io.snp_phased_ancestry(dest)
    handler.write(
        calldata=calldata,
        ancestries=lai,
        A=8,
        n_threads=os.cpu_count() // 2,
    )

    return calldata, lai


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