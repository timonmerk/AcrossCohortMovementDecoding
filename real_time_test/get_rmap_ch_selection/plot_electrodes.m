close all, clear all, clc

addpath C:\code\wjn_toolbox
addpath C:\code\spm12
addpath(genpath('C:\code\leaddbs'))

% spm('defaults','eeg')

%T = readtable('df_all_3_cohorts_custom.csv','Format','%f%s%s%s%f%f%f%f%f%f%f%f%f');
%T = readtable('df_ch_performances.csv');
T = readtable('electrodes_new_patient.csv');

%mni = [T.x_coord T.y_coord T.z_coord];
%md = T.mov_detection_rate_test;
%ba = T.balanced_acc_test	;

ba = [1, 1, 1, 1, 1, 1];

mni = [abs(T.x) T.y T.z];
ctx=export(gifti('BrainMesh_ICBM152Left_smoothed.gii'));

nmni=[];
for a=1:size(mni,1)
    [mind(1,a),i(1,a)] = min(wjn_distance(ctx.vertices,[-abs(mni(a,1)) mni(a,2:3)]));
    nmni(a,:) = ctx.vertices(i(a),:);
end

close all, 
figure('color','w')
p=wjn_plot_surface(ctx);
figone(40,40)
view(-90,15)
camlight 
material dull
% alpha 1
hold on
cm = colormap('viridis');  % viridis % jet
wjn_plot_colored_spheres(nmni,ba-0.5, 3,cm)  %wjn_gaussianize % change r to 2
%colorbar
hold on 
camzoom(2)
set(gcf,'color','none')
myprint('my_per_gp_2cm')
