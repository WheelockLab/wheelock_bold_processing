import argparse, logging, os, re, scipy, shlex, shutil
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd

from pathlib import Path
from subprocess import Popen, PIPE

PARCELLATION_LABELS = Path('/data/wheelock/data1/parcellations/333parcels/Parcels_LR.dlabel.nii')
OUTPUT_GRID = Path('/data/wheelock/data1/dierkerd/MNI152_T1_2mm.nii.gz')
PARCEL_NAME = 'Gordon333'
PARCELS = 333
PARCEL_PATH = Path('/data/wheelock/data1/parcellations/IM/Gordon_333_IM_with_roi_ordered_16networks.mat')
IM_333 = scipy.io.loadmat('/data/wheelock/data1/parcellations/IM/IM_Gordon_2016_333_Parcels_13nets.mat')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='BOLD processing that is more customizable than available pipelines. Based on the processing found in the ABCD HCP pipeline.')
    parser.add_argument('source_dir', help='Root directory for data to be processed', type=str)
    parser.add_argument('subject', help="The subject id to be processed (this does need to include the 'sub-') if it follows BIDS", type=str)
    parser.add_argument('results_path', help='Path to where results will be saved', type=str)
    parser.add_argument('-s', '--session_id', help="The session to be analyzed (this does need to include the 'ses-') if it follows BIDS", type=str)
    parser.add_argument('--settings', help='json file with settings locations. Use this if nibabies was NOT used for preprocessing', type=str)
    parser.add_argument('--save_figs', help='Save figures from analysis', action='store_true')
    parser.add_argument('--gsr', help='Include Global Signal Regression (gsr).', action='store_true')
    parser.add_argument('--fd_type', help='L1 (1) or L2 (2). Default: 1', type=int, default=1)
    parser.add_argument('--fd_thr', help='Threshold level for framewise displacement (default: 0.2)', type=float, default=0.2)
    parser.add_argument('--brain_radius', help='Estimate of brain radius in mm. Default: 50', type=int, default=50)
    parser.add_argument('--auto_radius', help="Calculate head radius using brain mask and assuming it's a sphere", action='store_true')
    parser.add_argument('--skip_sec', help='Seconds to skip at start of each functional run. Default: 0', type=float, default=0)
    parser.add_argument('--lp_hz', help='Low-frequency cut-off for filtering signal. Default: 0.009', type=float, default=0.009)
    parser.add_argument('--hp_hz', help='High-frequency cut-off for filtering signal. Default: 0.08', type=float, default=0.08)
    parser.add_argument('--bp_order', help='Butterworth filter order. Default: 2', type=int, default=2)
    return parser.parse_args()

def main(source_dir, subject, results_path, **kwargs):
    # Set all keyword variables
    bp_order = kwargs.get('bp_order')
    save_figs = True if 'save_figs' in kwargs and kwargs['save_figs'] else False
    gsr = True if 'gsr' in kwargs and kwargs['gsr'] else False
    fd_type = kwargs.get('fd_type')
    fd_max = kwargs.get('fd_thr')
    skip_seconds = kwargs.get('skip_sec')
    brain_radius = kwargs.get('brain_radius')
    auto_radius = True if 'auto_radius' in kwargs and kwargs['auto_radius'] else False
    session_id = kwargs['session_id'] if 'session_id' in kwargs else False
    lowpass_hz = kwargs.get('lp_hz')
    highpass_hz = kwargs.get('hp_hz')

    # Set up all output directories
    OUTPUT_DIR = Path(results_path)
    OUTPUT_DIR = OUTPUT_DIR / f'{subject}'
    
    if session_id:
        OUTPUT_DIR = OUTPUT_DIR / f'{session_id}'
    if gsr:
        OUTPUT_DIR = OUTPUT_DIR / 'gsr'
    else:
        OUTPUT_DIR = OUTPUT_DIR / 'no_gsr'
    # if OUTPUT_DIR.resolve().exists():
    #     shutil.rmtree(OUTPUT_DIR.resolve())
    OUTPUT_DIR.resolve().mkdir(parents=True, exist_ok=True)

    SOURCE_DIR = Path(source_dir)
    SUBJECT_DIR = SOURCE_DIR / f'{subject}'
    if session_id:
        SUBJECT_DIR = SUBJECT_DIR / f'{session_id}'
    SUBJECT_DIR = SUBJECT_DIR / 'func'
    cii_input_files = []
    for file_in in os.listdir(str(SUBJECT_DIR.resolve())):
        match_object = re.match(r'(sub-[0-9]{2,3})(_ses-[0-9]{1,2})?_(task-rest_run-[0-9]{1,2})(.*_bold)\.dtseries\.nii', file_in)
        # Match object groups:
        # Group 1: subject number
        # Group 2: session number
        # Group 3: run number
        # Group 4: prefix before 'bold'
        if match_object:
            cii_input_files.append(match_object)
        
    if len(cii_input_files) == 0:
        raise RuntimeError(f'No dtseries found for subject {subject}')

    # Set up loggers
    console_log_handler = logging.StreamHandler()
    log_file = f'{subject}'
    if session_id:
        log_file += f'_{session_id}'
    log_file += f'.log'
    file_log_handler = logging.FileHandler((OUTPUT_DIR / f'{log_file}').resolve(), mode='a', encoding='utf-8')
    logger.addHandler(console_log_handler)
    logger.addHandler(file_log_handler)

    # Start processing files
    cii_input_files.sort(key=lambda x: x.group(3))
    for cii_input in cii_input_files:
        settings = load_settings(source_dir, cii_input.group(1), cii_input.group(3), session_id=kwargs.get('session_id'), settings_file=kwargs.get('settings'))
        input_file = SUBJECT_DIR / f'{cii_input.group(0)}'
        logger.info(f'Processing cifti file: {input_file}')
        cii_image = nb.load(input_file.resolve())
        cii_data = cii_image.get_fdata()
        dvar_pre_reg = calculate_dvars_from_cifti(cii_data)

        if gsr and save_figs:
            logger.info(f'Global signal regression selected. Saving global signal trace')
            glob = np.concatenate((settings['white_matter'].to_numpy().reshape(-1, 1), settings['csf'].to_numpy().reshape(-1, 1), settings['global_signal'].to_numpy().reshape(-1, 1)), axis=1)
            d_glob = np.vstack(([0, 0, 0], np.diff(glob, axis=0)))
            plt.figure(figsize=(8, 4))
            plt.plot(glob, linewidth=1, label=('White Matter', 'CSF', 'Global Signal'))
            plt.title('Global signals')
            plt.legend()
            plt.savefig((OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_globalsignal_trace.png').resolve(), format='png', dpi=300)
            plt.close()

        TR = cii_image.header.get_axis(0).step # in whole seconds
        if TR > 20:
            TR /= 1000 # account for ms in some headers
        logger.info(f'TR: {TR} s')

        if auto_radius:
            brain_radius = calculate_brain_radius(SUBJECT_DIR, cii_input)
        frisson_regressors = make_friston_regressors(settings['movement_regressors'].to_numpy(), brain_radius)
        framewise_displacement, mean_framewise_displacement = calculate_framewise_displacement(
            frisson_regressors[:, :6], FD_type=fd_type
        )
        framewise_displacement_file_name = OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_framewise_displacement.txt'
        np.savetxt(framewise_displacement_file_name.resolve(), framewise_displacement)
        logger.info(f'Saving regressor file {framewise_displacement_file_name}')
        
        keepframes = np.full(len(framewise_displacement), True)
        if fd_max > 0:
            keepframes = framewise_displacement <= fd_max
        skip_frames = np.int8(np.floor(skip_seconds / TR))
        keepframes[:skip_frames] = False
        if cii_input == cii_input_files[0]:
            combined_framewise_displacement = framewise_displacement.copy()
            combined_keepframes = keepframes.copy()
        else:
            combined_framewise_displacement = np.hstack((combined_framewise_displacement, framewise_displacement))
            combined_keepframes = np.hstack((combined_keepframes, keepframes))

        if save_figs:
            plot_framewise_displacement(OUTPUT_DIR, TR, framewise_displacement, cii_input, keepframes)

        regressors = frisson_regressors # no GSR
        if gsr:
            glob = np.concatenate((settings['white_matter'].to_numpy().reshape(-1, 1), settings['csf'].to_numpy().reshape(-1, 1), settings['global_signal'].to_numpy().reshape(-1, 1)), axis=1)
            d_glob = np.vstack(([0, 0, 0], np.diff(glob, axis=0)))
            regressors = np.concatenate((glob, d_glob, frisson_regressors), axis=1) # GSR

        logger.info('Calculating regressors...')
        regressors = regressors - np.mean(regressors[keepframes, :], axis=0)
        regressors = detrend_manual(regressors, keepframes)
        detrend_data = cii_data - np.mean(cii_data[keepframes, :], axis=0)
        detrend_data = detrend_manual(detrend_data, keepframes)

        if save_figs:
            crange = [-200, 200] # using +/-2% as Power's papers, DCAN ABCD uses +/-6%
            plt.figure(figsize=(8, 4))
            im = plt.imshow(detrend_data.transpose(), aspect='auto', cmap='gray')
            plt.colorbar(location='right')
            im.set_clim(crange)
            plt.plot(np.where(keepframes == 0)[0], np.repeat(0, sum(keepframes == 0)), 'r|')
            plt.yticks([])
            plt.xlabel('TR')
            plt.savefig((OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_grayplots_all.png').resolve(), format='png', dpi=300)
            plt.close()

            plt.figure(figsize=(8, 4))
            X = detrend_data.copy()
            X[np.where(keepframes == 1)][0] = np.NaN
            im = plt.imshow(X.transpose(), aspect='auto', cmap='gray')
            plt.colorbar(location='right')
            im.set_clim(crange)
            plt.yticks([])
            plt.xlabel('TR')
            plt.savefig((OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_grayplots_removed.png').resolve(), format='png', dpi=300)
            plt.close()

            plt.figure(figsize=(8, 4))
            X = detrend_data.copy()
            X[np.where(keepframes == 0)][0] = np.NaN
            im = plt.imshow(X.transpose(), aspect='auto', cmap='gray')
            plt.colorbar(location='right')
            im.set_clim(crange)
            plt.yticks([])
            plt.xlabel('TR')
            plt.savefig((OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_grayplots_retained.png').resolve(), format='png', dpi=300)
            plt.close()

        b, _, _, _ = np.linalg.lstsq(regressors[keepframes, :], detrend_data[keepframes, :], rcond=None)
        data_post_regression = detrend_data - regressors @ b
        dvar_post_regression = calculate_dvars_from_cifti(data_post_regression)

        x = np.where(keepframes)[0]
        x_removed = np.where(keepframes == 0)[0]
        x_outside_bounds = (x_removed < x[0]) | (x_removed > x[-1])
        y_removed = np.apply_along_axis(lambda col: np.interp(x_removed, x, col[keepframes]), axis=0, arr=data_post_regression)
        y_mean = np.mean(data_post_regression[keepframes, :], axis=0)
        y_removed[x_outside_bounds, :] = y_mean

        logger.info(f'Interpolating frames with framewise displacement above {fd_max}')
        data_interpolated = data_post_regression.copy()
        data_interpolated[keepframes == 0, :] = y_removed

        sampling_freq = 1 / TR
        nyquist_freq = sampling_freq / 2
        butterworth_filter_numerator, butterworth_filter_denominator = scipy.signal.butter(bp_order / 2, np.array([lowpass_hz, highpass_hz]) / nyquist_freq, 'bandpass')

        padding = np.zeros_like(data_interpolated)
        padding_amount = padding.shape[0] # Rows to pad

        temp = np.vstack((padding, data_interpolated, padding))

        logger.info(f'Applying {lowpass_hz} - {highpass_hz} Hz filter to data')
        data_filtered = scipy.signal.filtfilt(butterworth_filter_numerator, butterworth_filter_denominator, temp, axis=0, padtype=None)
        data_filtered = data_filtered[padding_amount:-padding_amount]
        if cii_input == cii_input_files[0]:
            combined_data_filtered = data_filtered.copy()
            combined_header = [cii_image.header.get_axis(0), cii_image.header.get_axis(1)]
        else:
            combined_data_filtered = np.vstack((combined_data_filtered, data_filtered))
            combined_header[0].size += cii_image.header.get_axis(0).size
            
        dvar_post_filtering = calculate_dvars_from_cifti(data_filtered)

        if save_figs:
            plt.figure(figsize=(8, 4))
            plt.plot(dvar_pre_reg, linewidth=1, label='DVARS pre-regression', color='b')
            plt.plot(dvar_post_regression, linewidth=1, label='DVARS post-regression', color='r')
            plt.plot(dvar_post_filtering, linewidth=1, label='DVARS post filtered', color='g')
            plt.xlabel('TR')
            plt.legend()
            plt.savefig((OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_dvars_trace.png').resolve(), format='png', dpi=300)
            plt.close()

        ax1 = cii_image.header.get_axis(0)
        ax2 = cii_image.header.get_axis(1)
        header = (ax1, ax2)
        output_img = nb.cifti2.cifti2.Cifti2Image(np.single(data_filtered), header)
        output_img.to_filename((OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_{cii_input.group(4)}_postproc.dtseries.nii').resolve())        

        if cii_input == cii_input_files[0]:
            combined_filename = f'{cii_input.group(1)}_task-rest_run-combined_{cii_input.group(4)}'
            shutil.copy((OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_{cii_input.group(4)}_postproc.dtseries.nii').resolve(), (OUTPUT_DIR / f'{combined_filename}_postproc.dtseries.nii').resolve())
        else:
            concatanate_cmd = [
                '/usr/local/bin/wb_command',
                '-cifti-merge',
                (OUTPUT_DIR / f'{combined_filename}_postproc.dtseries.nii').resolve(),
                '-cifti',
                (OUTPUT_DIR / f'{combined_filename}_postproc.dtseries.nii').resolve(),
                '-cifti',
                (OUTPUT_DIR / f'{cii_input.group(1)}_{cii_input.group(3)}_{cii_input.group(4)}_postproc.dtseries.nii').resolve()
            ]
            with Popen(concatanate_cmd, stdout=PIPE, stderr=PIPE) as p:
                while p.poll() is None:
                    for line in p.stdout:
                        print(line.decode(), end='')
                if p.poll() != 0:
                    raise RuntimeError('Error during concatating runs')
        
    np.savetxt((OUTPUT_DIR / f'{combined_filename}_combined_framewise_displacement.txt').resolve(), combined_framewise_displacement)
    if save_figs:
        plot_framewise_displacement(OUTPUT_DIR, TR, combined_framewise_displacement, cii_input, combined_keepframes, combined=True)

    parcellate_data(OUTPUT_DIR)

def load_settings(source_dir, subject, run, session_id=None, settings_file=None):
    settings = {}
    
    out_suffix = f'{subject}'
    if session_id is not None:
        out_suffix = f'{out_suffix}/{session_id}'

    settings['out_suffix'] = out_suffix
    if settings_file is None:
        confounds_file = f'{source_dir}/{subject}'
        if session_id is not None:
            confounds_file += f'/{session_id}'
        confounds_file += f'/func/{subject}'
        if session_id is not None:
            confounds_file += f'_{session_id}'
        confounds_file += f'_{run}_desc-confounds_timeseries.tsv'
        settings.update(load_nibabies_confounds(confounds_file))
    else:
        raise Exception("This feature has no been implemented")

    return settings

def load_nibabies_confounds(confounds_file):
    confounds = pd.read_csv(confounds_file, delimiter='\t', header=0)
    return {
        'movement_regressors': confounds[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']],
        'white_matter': confounds['white_matter'],
        'csf': confounds['csf'],
        'global_signal': confounds['global_signal']
    }
    
def calculate_dvars_from_cifti(data):
    """
    This calculate dvars (derivative of variance) based on grayordinates

    Args:
        data (2D array): 2D array with shape (Tp, g): Tp = timepoints, g = number of grayordinates

    Returns:
        dvars (float): calculated dvars value
    """
    # Format data so rows and columns are consistent
    number_timepoints, number_grayordinates = data.shape
    if number_grayordinates < number_timepoints:
        data = data.T
        print("Data transposed because timepoints > grayordinates")

    # Calculate differences across timepoints
    data_difference = np.diff(data, axis=0)

    # Calculate dvars as rms of differences
    return np.hstack((np.nan, np.sqrt(np.mean(data_difference**2, axis=1))))

def make_friston_regressors(R, hd_mm):
    """
    This function takes a matrix `MR` of 6 degrees of freedom (DOF) movement correction
    parameters and calculates the corresponding 24 Friston regressors.

    Args:
        MR : numpy array of shape (r, c)
            A matrix where r is the number of time points and c are the 6 DOF movement regressors.
            If the number of columns is more than 6, only the first 6 columns are considered.

        hd_mm : float, optional
            The head radius in mm. Default is 50 mm.

    Returns:
        FR : numpy array of shape (r, 24)
            A matrix containing 24 Friston regressors.
    """
    MR = R[:, :6]
    MR[:, 3:] = MR[:, 3:] * np.pi * hd_mm / 180
    # Calculate the first part of the Friston regressors (MR and MR^2)
    FR = np.hstack([MR, MR**2])

    # Create a dummy array for the temporal derivatives (lagged version of FR)
    dummy = np.zeros_like(FR)
    dummy[1:, :] = FR[:-1, :]  # shift FR by one time step
    dummy[0, :] = 0  # set the first row to 0

    # Concatenate the original FR and the lagged version
    FR = np.hstack([FR, dummy])
    return FR

def calculate_framewise_displacement(regressors, FD_type=1):
    '''
    This function calculates framewise displacement (Power et al. 2012 Neuroimage)
    The columns 3-6 (angular displacement) is assumed to be already converted to mm before passed in this function

    Args:
        regressors: movement regrossors

    Returns:
        framewise_displacement: framewise displacement
        mean_framewise_displacement: mean framewise displacement
    '''

    dregressors = np.diff(regressors, axis=0)  # First-order derivative
    ddregressors = np.diff(dregressors, axis=0)  # Second-order derivative
    if FD_type == 1:
        # L1-norm - sum of absolute values of first-order derivatives
        framewise_displacement = np.sum(np.absolute(dregressors), axis=1)
        mean_framewise_displacement = np.mean(framewise_displacement)
        framewise_displacement = np.hstack((np.zeros(1), framewise_displacement))  # Pad zeros to make it the same length as the original data
    elif FD_type == 2:
        # L2-norm - sum of absolute values of second-order derivatives
        framewise_displacement = np.sum(np.absolute(ddregressors), axis=1)
        mean_framewise_displacement = np.mean(framewise_displacement)
        framewise_displacement = np.hstack((np.zeros(2), framewise_displacement)) # Pad zeros to make it the same length as the original data
    return framewise_displacement, mean_framewise_displacement

def detrend_manual(data, keepframe):
    '''
    Remove linear trends
    '''
    detrended_data = data.copy()
    time_points = np.where(keepframe)[0]
    time_points_all = np.array(range(keepframe.shape[0]))

    # Create the design matrix for linear regression (constant + linear term)
    X = np.vstack(
        [time_points, np.ones(len(time_points))]
    ).T  # Shape (len(keepframe), 2)
    Xall = np.vstack([time_points_all, np.ones(len(time_points_all))]).T

    # Perform the linear regression for all columns at once using least squares
    # Y is the data[keepframe, :] with shape (len(keepframe), n)
    Y = data[keepframe, :]

    # Compute the least squares solution to find the slope and intercept for each column
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # beta shape is (2, n)

    # Calculate the trend for each column using the coefficients
    trend = Xall @ beta  # Shape (len(keepframe), n)

    # Subtract the trend from the data at the all indices
    detrended_data -= trend

    return detrended_data

def parcellate_data(data_path):
    if isinstance(data_path, str):
        data_path = Path(data_path)

    dtseries_files = data_path.glob('*.dtseries.nii')
    ptseries_files = []
    for dtseries_file in dtseries_files:
        name_match = re.match(r'(sub-[0-9]{2,3})(_ses-[0-9]{1,2})?(_task-rest_run-[0-9]{1,2})?(.*_postproc)\.dtseries\.nii', dtseries_file.name)
        ptseries_name = f'{name_match.group(1)}'
        if name_match.group(2) is not None and name_match.group(2):
            ptseries_name += f'{name_match.group(2)}'
        if name_match.group(3) is not None and name_match.group(3):
            ptseries_name += f'{name_match.group(3)}'
        ptseries_name += f'_{PARCEL_NAME}.ptseries.nii'
        ptseries_path = data_path / f'{ptseries_name}'
        ptseries_files.append(ptseries_path)

        parcellate_command = f'/usr/local/bin/wb_command -cifti-parcellate {dtseries_file} {PARCELLATION_LABELS} COLUMN {ptseries_path}'
        parcellate_command_split = shlex.split(parcellate_command)
        with Popen(parcellate_command_split, stdout=PIPE, stderr=PIPE) as p:
            print('Applying parcellation...')
            while p.poll() is None:
                for line in p.stdout:
                    print(line.decode(), end='')
            if p.poll() != 0:
                raise RuntimeError('Error during parcellation step')

    return ptseries_files

def create_functional_connectivity(parcellated_data_files, save_figs=False):
    for data_file in parcellated_data_files:
        ptseries_loaded = nb.load(data_file.resolve())
        ptseries_data = ptseries_loaded.get_fdata()
        correlation = np.corrcoef(ptseries_data)

        if save_figs:
            IM_333 = IM_333.IM
            plt.figure(figsize=(8, 4))
            
def plot_framewise_displacement(output_dir, tr, framewise_displacement, cii_input_match, keepframes, combined=False):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes()
    ticks = [x for x in range(len(framewise_displacement))]
    tick_labels = [x * tr for x in range(len(framewise_displacement))]
    for j in np.where(keepframes == 0)[0]:
        plt.axvline(x=j, color=[0.5, 0.5, 0.5], alpha=0.5)
    plt.plot(ticks, framewise_displacement, linewidth=1)
    ax.set_xlim(left=0, right=ticks[-1])
    plt.xticks(np.arange(0, ticks[-1], step=100), np.arange(0, tick_labels[-1], step=100 * tr))
    plt.title(f'Framewise Displacement\nTotal Time: {int(tr * len(keepframes))} seconds Usable Time: {int(tr * len(np.where(keepframes == True)[0]))} seconds {int(100 * len(np.where(keepframes == True)[0]) / len(keepframes))}%')
    plt.xlabel('Time (s)')
    save_fig_name = output_dir / f'{cii_input_match.group(1)}_{cii_input_match.group(3)}_fd_trace.png'
    if combined:
        save_fig_name = output_dir / f'{cii_input_match.group(1)}_combined_fd_trace.png'
    plt.savefig(save_fig_name.resolve(), format='png', dpi=300)
    plt.close()

def calculate_brain_radius(source_dir, cii_input):
    brain_mask_files = source_dir.glob(f'{cii_input.group(1)}_*_{cii_input.group(3)}_desc-brain_mask.nii.gz')
    if len(brain_mask_files) >= 1:
        brain_mask_file = brain_mask_files[0]
    else:
        raise RuntimeError(f'No brain mask found for {cii_input.group(1)} {cii_input.group(3)}')

    brain_mask = nb.load(source_dir / f'{brain_mask_file}')
    brain_mask_data = brain_mask.get_fdata()
    voxel_volume = np.prod(brain_mask.header.get_zooms())
    total_mask_voxels = np.sum(brain_mask_data)
    total_mask_volume = voxel_volume * total_mask_voxels

    brain_radius = ((3 * total_mask_volume) / (4 * np.pi)) ** (1 / 3)
    logger.info(f'{cii_input.group(1)} {cii_input.group(3)} brain radius estimated to be {brain_radius} mm')
    return brain_radius
    
    
def matrix_org3(figure_object, data_matrix, key, buffer, limits, color_map, colormap_data):
    dim_y, dim_x = data_matrix.shape
    if dim_y == dim_x:
        type = 'square'

    figure_object.imshow(data_matrix, colormap='jet', vmin=limits[0], vmax=limits[1], extent=(0.5 - buffer, dim_x + 0.5, 0.5, dim_y + buffer + 0.5))

    networks = list(set(key[0][0][:,1]))
    key_vals = key[0][0]
    number_networks = len(networks)

    if type == 'square':
        for network1 in range(number_networks):
            if any(np.argwhere(key_vals[:,1] == networks[network1])):
                plt.plot([0.5, dim_x + 0.5], [np.argwhere(key_vals[:,1] == networks[network1])[-1] + 0.5, np.argwhere(key_vals[:,1] == networks[network1])[-1] + 0.5], 'k')
                plt.plot([np.argwhere(key_vals[:,1] == networks[network1])[-1] + 0.5, np.argwhere(key_vals[:,1] == networks[network1])[-1] + 0.5], [0.5, dim_x + 0.5], 'k')

        for network2 in range(number_networks):
            if any(np.argwhere(key_vals[:,1] == networks[network2])):
                x = 0.5 - buffer
                y = np.argwhere(key_vals[:,1] == networks[network2])[0] - 0.5
                w = buffer
                h = np.argwhere(key_vals[:,1] == networks[network2])[-1] - y + 0.5
                figure_object.Rectangle((x, y), w, h, fc=color_map[networks[network2], :])

                x = y
                w = np.argwhere(key_vals[:,1] == networks[network2])[-1] - x + 0.5
                y = dim_y + 0.5
                h = buffer
                figure_object.Rectangle()
           
    else:
        raise Exception("This feature has not been implemented")
    


if __name__ == '__main__':
    main(**vars(parse_args()))