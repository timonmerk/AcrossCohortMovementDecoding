%addpath C:\code\wjn_toolbox
addpath C:\code\wjn_toolbox-master
addpath C:\code\spm12
spm('defaults','eeg')
%%
load ECoG_Atlas_Master_Table_New.mat
% load ECoG_Atlas_Master_Table.mat
itheta = find(data_T.Freqband==1); 
ialpha = find(data_T.Freqband==2);
ibeta = find(data_T.Freqband==3);
igamma = find(data_T.Freqband==6);
% mni = [abs(data_T.Channel_mni_X) data_T.Channel_mni_Y data_T.Channel_mni_Z];
mni = [data_T.Channel_mni_X data_T.Channel_mni_Y data_T.Channel_mni_Z];
% cm = wjn_erc_colormap;
cm = colormap('viridis');  % viridis % jet


beta_ = data_T.beta_maximum_peak_amplitude;



close all, 
figure
wjn_plot_surface('C:\code\leaddbs\templates\space\MNI_ICBM_2009b_NLIN_ASYM\cortex\CortexLowRes_15000V.mat')

alpha 0.3
hold on
p=wjn_plot_colored_spheres(mni(ibeta,:),beta_(data_T.Freqband==3),2,cm);
caxis([-1.5, 4.2])
colorbar

p=wjn_plot_colored_spheres(mni(itheta,:),[],2,cm(1,:));
p=wjn_plot_colored_spheres(mni(ialpha,:),[],2,cm(2,:));

p=wjn_plot_colored_spheres(mni(igamma,:),[],2,cm(4,:));


% close all, 
% figure
% wjn_plot_surface('C:\code\leaddbs\templates\space\MNI_ICBM_2009b_NLIN_ASYM\cortex\CortexLowRes_15000V.mat',[data_T.beta_total_integral mni])
% alpha 0.3
% hold on
% p=wjn_plot_colored_spheres(mni(itheta,:),[],2,cm(1,:));
% p=wjn_plot_colored_spheres(mni(ialpha,:),[],2,cm(2,:));
% p=wjn_plot_colored_spheres(mni(ibeta,:),[],2,cm(3,:));
% p=wjn_plot_colored_spheres(mni(igamma,:),[],2,cm(4,:));
% % camlight