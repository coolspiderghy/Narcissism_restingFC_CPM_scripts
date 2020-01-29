
clear;
clc;

%load FC_matrix.mat;
%load NPI.mat;
load Group_swcar_network_264ROIs_wm_csf_mvmnt_eigen1.mat
load TAI_BDI.mat
%all_mats  = FC_matrix;
all_mats = GroupMatrix;
%all_behav = NPI.data(:,strcmp('NPI_score',NPI.label));
all_behav = TAI_BDI(:,1);
% threshold for feature selection
thresh = 0.005;
Pre_Method = 'Normalize';
out_name='UCLA_lone';
corr_type='Pearson';

%% 2 folds
for mm=1:100
    mkdir([pwd,'\N_twofolds\Times_' num2str(mm)]);
    outpath = [pwd,'\N_twofolds\Times_' num2str(mm)];
    predfunc_Nfolds(all_mats,all_behav,thresh,Pre_Method,outpath,2,out_name,corr_type);
end

% do the permutation
mkdir([pwd,'\N_twofolds_Permutation_' num2str(thresh)]);
outpath = [pwd,'\N_twofolds_Permutation_' num2str(thresh)];
for it = 1:1000 % 1000 times
    fprintf('\n Performing iteration %d out of %d', it, 1000);
    if it <10
        mkdir([outpath '\Times_00' num2str(it)]);
        newpath = [outpath '\Times_00' num2str(it)];
    elseif it > 9 & it < 100
        mkdir([outpath '\Times_0' num2str(it)]);
        newpath = [outpath '\Times_0' num2str(it)];
    else
        mkdir([outpath '\Times_' num2str(it)]);
        newpath = [outpath '\Times_' num2str(it)];
    end
    
    new_behav = all_behav(randperm(length(all_behav)));
    predfunc_Nfolds(all_mats,new_behav,thresh,Pre_Method,newpath,2,out_name,corr_type);
end




%% LOOCV
outcome = predfunc_LOSOCV(all_mats,all_behav,thresh,Pre_Method,pwd,out_name,corr_type);
% do the permutation
mkdir([pwd,'\Permutation_motion_' num2str(thresh)]);
outpath = [pwd,'\Permutation_motion_' num2str(thresh)];

for it = 1:1000 % 1000 times
    fprintf('\n Performing iteration %d out of %d', it, 1000);
    if it <10
        mkdir([outpath '\Times_00' num2str(it)]);
        newpath = [outpath '\Times_00' num2str(it)];
    elseif it > 9 & it < 100
        mkdir([outpath '\Times_0' num2str(it)]);
        newpath = [outpath '\Times_0' num2str(it)];
    else
        mkdir([outpath '\Times_' num2str(it)]);
        newpath = [outpath '\Times_' num2str(it)];
    end

    new_behav = all_behav(randperm(length(all_behav)));
    outcome = predfunc_LOSOCV_motion(all_mats,new_behav,thresh,Pre_Method,newpath,out_name,corr_type);
end






