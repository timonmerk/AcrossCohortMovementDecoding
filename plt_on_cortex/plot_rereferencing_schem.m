close all, clear all, clc

addpath C:\code\wjn_toolbox
addpath C:\code\spm12
addpath(genpath('C:\code\leaddbs'))

% spm('defaults','eeg')

%T = readtable('df_all_3_cohorts_custom.csv','Format','%f%s%s%s%f%f%f%f%f%f%f%f%f');
%T = readtable('df_ch_performances.csv');
T = readtable('df_ch_performances.csv');

%mni = [T.x_coord T.y_coord T.z_coord];
%md = T.mov_detection_rate_test;
%ba = T.balanced_acc_test	;

ba = T.performance_test;

% there are NaN's
ba(isnan(ba))=0.5;

mni = [abs(T.x) T.y T.z];
ctx=export(gifti('BrainMesh_ICBM152Left_smoothed.gii'));

mni_coords = mni(82:87, :);

nmni=[];
for a=1:size(mni_coords,1)
    [mind(1,a),i(1,a)] = min( ...
        wjn_distance( ...
        ctx.vertices, ...
        [-abs(mni_coords(a,1)) mni_coords(a,2:3)] ...
        ) ...
      );
    nmni(a,:) = ctx.vertices(i(a),:);
end

% Bipolar Rereferencing
c_vec = [0, 0, 0.25, 0.5, 0.25, 0];
c_vec_black = [0, 0, 0, 0, 0, 0]+1;

close all, 
figure('color','w')
p=wjn_plot_surface(ctx);
figone(40,40)
view(-90,15)
camlight 
material dull
% alpha 1
hold on
cm = colormap('gray');  % viridis % jet
wjn_plot_colored_spheres(nmni, c_vec_black, 6,cm)  %wjn_gaussianize % change r to 2
cm = colormap('viridis');  % viridis % jet
wjn_plot_colored_spheres(nmni(3:4, :), c_vec(3:4), 6,cm)  %wjn_gaussianize % change r to 2
%colorbar
hold on 
camzoom(2)
set(gcf,'color','none')
myprint('bipolar_reref')

% Common Average Rereferencing
c_vec = [0.39, 0.31, 0.25, 0.5, 0.1, 0.23];
c_vec_black = [0, 0, 0, 0, 0, 0]+1;

close all, 
figure('color','w')
p=wjn_plot_surface(ctx);
figone(40,40)
view(-90,15)
camlight 
material dull
% alpha 1
hold on
cm = colormap('gray');  % viridis % jet
wjn_plot_colored_spheres(nmni, c_vec_black, 6,cm)  %wjn_gaussianize % change r to 2
cm = colormap('viridis');  % viridis % jet
wjn_plot_colored_spheres(nmni, c_vec, 6,cm)  %wjn_gaussianize % change r to 2
%colorbar
hold on 
camzoom(2)
set(gcf,'color','none')
myprint('bipolar_reref')

% No Rereferencing
c_vec = [0, 0, 0.25, 0.5, 0.25, 0];
c_vec_black = [0, 0, 0, 0, 0, 0]+1;

close all, 
figure('color','w')
p=wjn_plot_surface(ctx);
figone(40,40)
view(-90,15)
camlight 
material dull
% alpha 1
hold on
cm = colormap('gray');  % viridis % jet
wjn_plot_colored_spheres(nmni, c_vec_black, 6,cm)  %wjn_gaussianize % change r to 2
cm = colormap('viridis');  % viridis % jet
wjn_plot_colored_spheres(nmni(4, :), c_vec(4), 6,cm)  %wjn_gaussianize % change r to 2

%colorbar
hold on 
camzoom(2)
set(gcf,'color','none')
myprint('bipolar_reref')