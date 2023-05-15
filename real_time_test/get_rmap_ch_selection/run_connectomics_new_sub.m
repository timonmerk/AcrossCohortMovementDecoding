addpath('C:\code\wjn_toolbox');
addpath(genpath('C:\code\leaddbs'));
addpath(genpath('C:\code\spm12'));

folder_out = "C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\real_time_test\get_rmap_ch_selection\OUT_func_conn";
csvfile = "electrodes_new_patient.csv";

T=readtable(csvfile);

for a =1:size(T,1)
    roiname  = fullfile(folder_out, strcat('Berlin_sub-', string(T.sub(a)), '_ROI_', string(T.ch{a}), '.nii'));
    mni = [T.x(a) T.y(a) T.z(a)];
    wjn_spherical_roi(roiname,mni,4);
end

% Now start LeadMapper to estimate the functional and structural connectivity fingerprints
% Next: trd_connectivity.py

