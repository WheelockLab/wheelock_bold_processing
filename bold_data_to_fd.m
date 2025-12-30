clear;
close all;
clc;

data_path = '/data/wheelock/data1/datasets/infant_eeg_fmri/pilot/bids/derivatives/bold_proc';
subject_num = 'sub-015';
ses_num = 'ses-01';
gsr = 'gsr';
data_path = strcat(data_path, '/', subject_num, '/', ses_num, '/', gsr);
out_dir = data_path;

% === Parcellate the ptseries with 333 atlas ===
format compact;
addpath(genpath('/data/wheelock/data1/software/cifti-matlab-master'));
addpath(genpath('/data/wheelock/data1/NLA/NLApublic/NLA_toolbox_070319/NLA_toolbox_v1.0/visualization'));
IM_333 = load('/data/wheelock/data1/parcellations/IM/IM_Gordon_2016_333_Parcels_13nets.mat').IM;
% Provide parcellation 333 atlas info
dlabel_333_path = '/data/wheelock/data1/parcellations/333parcels/Parcels_LR.dlabel.nii';

% out_ptseries_path = strcat(outdir,'/333_parcels_ptseries');
% mkdir(out_ptseries_path);

% Provide parcellation 333 atlas info
dlabel_333_path = '/data/wheelock/data1/parcellations/333parcels/Parcels_LR.dlabel.nii';

% Provide parcellation subcortical 8 atlas
SourceDir = '/data/wheelock/data1/datasets/infant_eeg_fmri/pilot/bids/derivatives/abcd_hcp_pipeline/';
OutputGrid = '/data/wheelock/data1/people/dierkerd/MNI152_T1_2mm.nii.gz';

% Provide combination 341 atlas info
parcelname = 'Gordon333_Desikan8Subcortical';
parcelnum = 333;
parcelpath = '/data/wheelock/data1/parcellations/IM/Gordon_333_IM_with_roi_ordered_16networks.mat';

files = dir(fullfile(data_path, '*dtseries.nii'));
files = files(~ismember({files.name}, {'.', '..'}));
file_paths = fullfile({files.folder}, {files.name});

for file = 1:length(file_paths)
    dtseries = file_paths{file};

    tokens = regexp(dtseries, 'sub-(\d+).*(run-(\d+)|combined)', 'tokens');

    if ~isempty(tokens);
        run_num = tokens{1}{2};
    end

    ptseries = strcat(out_dir, '/', subject_num, '_', run_num, '_Gordon333.ptseries.nii')

    parcellate_command = sprintf('/usr/local/bin/wb_command -cifti-parcellate %s %s COLUMN %s', dtseries, dlabel_333_path, ptseries);
    [status1, cmdout1] = system(parcellate_command);
    status1
    cmdout1

    ptseries_data = cifti_read(ptseries);
    pt_333 = ptseries_data.cdata;
    pt_333 = pt_333';
    corr_mat = corrcoef(pt_333);

    figure('Color', 'w');
    Matrix_Org3(corr_mat(IM_333.order, IM_333.order), IM_333.key, 10, [-0.3, 0.5], IM_333.cMap, 0), colorbar;
    plot_title = strcat(subject_num, " ", run_num, ' Functional Connectivity');
    if isequal(gsr, 'gsr')
        plot_title = strcat(plot_title, ' GSR');
    else
        plot_title = strcat(plot_title, ' No GSR');
    end
    title(plot_title);
    saveas(gcf, strcat(out_dir, '/', subject_num, '_', run_num, '_', gsr, '_functional_conn.png'));
end