close all, clear all, clc
addpath C:\code\wjn_toolbox
addpath C:\code\spm12
addpath(genpath('C:\code\leaddbs'))
spm('defaults','eeg')


T = readtable('ECoG Coordinates Bipolar.csv');
mni = [T.Var2 T.Var3 T.Var4];
ctx=export(gifti('BrainMesh_ICBM152Left_smoothed.gii'));

nmni=[];
for a=1:size(mni,1)
    [mind(1,a),i(1,a)] = min(wjn_distance(ctx.vertices,[-abs(mni(a,1)) mni(a,2:3)]));
    nmni(a,:) = ctx.vertices(i(a),:);
end

t = readtable('Automated Anatomical Labeling 3 (Rolls 2020).txt','Delimiter',' ');
roi = {'Parietal','Frontal','Postcentral_L','Precentral_L'};
nii = ea_load_nii('Automated Anatomical Labeling 3 (Rolls 2020).nii');

i = ci(roi,t.Var2);
ccc= wjn_erc_colormap;
ccc = ccc([3 4 1 2 5:end],:);
cc = repmat(ccc(5,:),length(ctx.vertices),1);
for a = 1:length(roi)
    i = ci(roi{a},t.Var2);
    ix=[];
    for b = 1:length(i)
        ix = [ix;find(nii.img(:)==i(b))];
    end
    ix = unique(ix);
    ixx = [];
    for b = 1:length(ix)
         [x,y,z]=ind2sub(size(nii.img),ix(b));
        loc = wjn_cor2mni([x,y,z],nii.mat);
        ixx = [ixx;find(wjn_distance(ctx.vertices,loc)<2)];
    end
       
    cc(unique(ixx),:) = repmat(ccc(a,:),[length(unique(ixx)) 1]);
end

close all, 
figure('color','w')
p=wjn_plot_surface(ctx,ccc(5,:));
p.FaceVertexCData = cc;
figone(40,40)
view(-90,15)
camlight 
hold on
cm = colormap('jet');
for a=1:5:31
wjn_plot_colored_spheres(nmni(a:a+4,:),[],2,[.25 .25 .25]+(a/5)/10)
end
material dull
hold on 
camzoom(3)
set(gcf,'color','none')
myprint('cortical_areas_spheres')
