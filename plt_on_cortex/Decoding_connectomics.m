close all, clear all, clc
addpath C:\code\wjn_toolbox
addpath C:\code\spm12
addpath(genpath('C:\code\leaddbs'))
spm('defaults','eeg')


T = readtable('df_all_3_cohorts_custom.csv','Format','%f%s%s%s%f%f%f%f%f%f%f%f%f');
mni = [T.x_coord T.y_coord T.z_coord];
md = T.mov_detection_rate_test;
ba = T.balanced_acc_test	;
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
cm = colormap('jet');
wjn_plot_colored_spheres(nmni,wjn_gaussianize(ba),1,cm)
hold on 
camzoom(3)
set(gcf,'color','none')
myprint('jet_gaussianized')

%%


%% dMRI
close all, 
% Manually smoothed all files with a 16x16x16 mm gaussian kernel.
[fnames,~,files] = ffind('../structural_connectivity/s_*.nii');
i=ci('Berlin_002',fnames);
fnames(i)=[];files(i)=[];
ss={};
for a = 1:size(T,1)
    subnum =  str2num(T.subject{a});
    if ~isempty(subnum)
        ss{a,1} = strcat(T.cohort{a},'_',num2str(subnum,'%03.f'),'_',T.ch_name{a});
    else
        ss{a,1} = strcat(T.cohort{a},'_',T.subject{a},'_',T.ch_name{a});
    end
end
y = wjn_gaussianize(ba);
M=[];ny=[];

for a = 1:length(files)
    fname = strsplit(fnames{a},'_');
    i = ci(strcat(fname{2},'_',fname{3},'_',fname{5},'_',fname{6},'_',fname{7},'_',fname{8}),ss);
    if isempty(i)
        i = ci(strcat(fname{2},'_',fname{3},'_',fname{5},'_',fname{6},'_',fname{7}),ss);
    end
    ny(a,1) = y(i);
       nii = ea_load_nii(files{a});
    if T.x_coord>0
        nii.img = nii.img(end:-1:1,:,:);
    end
    M(:,a) = nii.img(:);
   
end
M(M==0)=nan;
r = corr(M',ny,'type','spearman','rows','pairwise');
nii.fname = 'dMRI_Rmap_mirrored.nii';
nii.img(:) = r(:);
ea_write_nii(nii)

spm_imcalc({'t1.nii','dMRI_Rmap_mirrored.nii'},'dMRI_R_hd.nii','i2')
spm_imcalc({'dMRI_R_hd.nii','gm_mask.nii'},'dMRI_mask.nii','i1.*(i2>0)')
spm_smooth('dMRI_mask.nii','s4dMRI_mask.nii',[4 4 4])

wjn_nii_average(files,'avg_nii.nii')

%% fMRI
close all, 
[fnames,~,files] = ffind('../functional_connectivity/*Fz.nii');
i=ci('Berlin_002',fnames);
fnames(i)=[];files(i)=[];
ss={};
for a = 1:size(T,1)
    subnum =  str2num(T.subject{a});
    if ~isempty(subnum)
        ss{a,1} = strcat(T.cohort{a},'_',num2str(subnum,'%03.f'),'_',T.ch_name{a});
    else
        ss{a,1} = strcat(T.cohort{a},'_',T.subject{a},'_',T.ch_name{a});
    end
end
y = wjn_gaussianize(ba);
M=[];ny=[];

for a = 1:length(files)
    fname = strsplit(fnames{a},'_');
    i = ci(strcat(fname{1},'_',fname{2},'_',fname{4},'_',fname{5},'_',fname{6},'_',fname{7}),ss);
    if isempty(i)
        i = ci(strcat(fname{1},'_',fname{2},'_',fname{4},'_',fname{5},'_',fname{6}),ss);
    end
    ny(a,1) = y(i);
       nii = ea_load_nii(files{a});
    if T.x_coord>0
        nii.img = nii.img(end:-1:1,:,:);
    end
    M(:,a) = nii.img(:);
   
end
M(M==0)=nan;
r = corr(M',ny,'type','spearman','rows','pairwise');
nii.fname = 'fMRI_Rmap_mirrored.nii';
nii.img(:) = r(:);
ea_write_nii(nii)

%% ARCHIVE
%% Viridis gaussianized

close all, 
figure('color','w')
p=wjn_plot_surface(ctx);
figone(40,40)
view(-90,15)
camlight 
material dull
% alpha 1
hold on
cm = colormap('viridis');
wjn_plot_colored_spheres(nmni,wjn_gaussianize(ba),1,cm)
hold on 
camzoom(3)
set(gcf,'color','none')
myprint('viridis_gaussianized')

%% Viridis 

close all, 
figure('color','w')
p=wjn_plot_surface(ctx);
figone(40,40)
view(-90,15)
camlight 
material dull
% alpha 1
hold on
cm = colormap('viridis');
wjn_plot_colored_spheres(nmni,ba,1,cm)
hold on 
camzoom(3)
set(gcf,'color','none')
myprint('viridis')


%% Jet

close all, 
figure('color','w')
p=wjn_plot_surface(ctx);
figone(40,40)
view(-90,15)
camlight 
material dull
% alpha 1
hold on
cm = colormap('jet');
wjn_plot_colored_spheres(nmni,ba,1,cm)
hold on 
camzoom(3)
set(gcf,'color','none')
myprint('jet')

