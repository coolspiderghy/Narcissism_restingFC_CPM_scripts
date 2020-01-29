function outcome = predfunc_LOSOCV_covariate(all_mats,all_behav,thresh,Pre_Method,outpath,covar,covar_name,covar_type,out_name,corr_type)
% Copyright 2015 Xilin Shen and Emily Finn

% This code is released under the terms of the GNU GPL v2. This code
% is not FDA approved for clinical use; it is provided
% freely for research purposes. If using this in a publication
% please reference this properly as:

% Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang, Chun MM,
% Papademetris X & Constable RT. (2015). Functional connectome
% fingerprinting: Identifying individuals using patterns of brain
% connectivity. Nature Neuroscience 18, 1664-1671.

% This code provides a framework for implementing functional
% connectivity-based behavioral prediction in a leave-one-subject-out
% cross-validation scheme, as described in Finn, Shen et al 2015 (see above
% for full reference). The first input ('all_mats') is a pre-calculated
% MxMxN matrix containing all individual-subject connectivity matrices,
% where M = number of nodes in the chosen brain atlas and N = number of
% subjects. Each element (i,j,k) in these matrices represents the
% correlation between the BOLD timecourses of nodes i and j in subject k
% during a single fMRI session. The second input ('all_behav') is the
% Nx1 vector of scores for the behavior of interest for all subjects.

% As in the reference paper, the predictive power of the model is assessed
% via correlation between predicted and observed scores across all
% subjects. Note that this assumes normal or near-normal distributions for
% both vectors, and does not assess absolute accuracy of predictions (only
% relative accuracy within the sample). It is recommended to explore
% additional/alternative metrics for assessing predictive power, such as
% prediction error sum of squares or prediction r^2.



% ---------------------------------------

no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred_all = zeros(no_sub,1);

for leftout = 1:no_sub;
    %fprintf('\n Leaving out subj # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3)); %from 3-dimension to 2-dimension
    
    train_behav = all_behav;
    train_behav(leftout) = [];
    
    train_covar = covar;
    train_covar(leftout) = [];
    
    test_mat = all_mats(:,:,leftout);
    
    %% set up pre methods
    if strcmp(Pre_Method, 'Normalize')
        %Normalizing
        train_vcts = train_vcts';
        MeanValue = mean(train_vcts);
        StandardDeviation = std(train_vcts);
        [~, columns_quantity] = size(train_vcts);
        for j = 1:columns_quantity
            train_vcts(:, j) = (train_vcts(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
        train_vcts = train_vcts';
        train_mats = reshape(train_vcts,size(train_mats,1),size(train_mats,2),size(train_mats,3));
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        train_vcts = train_vcts';
        MinValue = min(train_vcts);
        MaxValue = max(train_vcts);
        [~, columns_quantity] = size(train_vcts);
        for j = 1:columns_quantity
            train_vcts(:, j) = (train_vcts(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
        train_vcts = train_vcts';
        train_mats = reshape(train_vcts,size(train_mats,1),size(train_mats,2),size(train_mats,3));
    end
    
    % Normalize test data
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        [ms,ns] = size(test_mat);
        test_mat = reshape(test_mat, 1,ms*ns);
        test_mat = (test_mat - MeanValue) ./ StandardDeviation;
        test_mat = reshape(test_mat, ms,ns);
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        [ms,ns] = size(test_mat);
        test_mat = reshape(test_mat, 1,ms*ns);
        test_mat = (test_mat - MinValue) ./ (MaxValue - MinValue);
        test_mat = reshape(test_mat, ms,ns);
    end
    
    
    
    
    % correlate all edges with behavior
    if covar_type==1 %if it is a continuous variable
        [r_mat,p_mat] = partialcorr(train_vcts',train_behav,train_covar,'type',corr_type); %correlate each edge with behavior
    else  %if it is a categorical variable
        tv_trans = train_vcts';
        for mmmm = 1:size(tv_trans,2)
            [~,~,ry(:,mmmm)] = regress(tv_trans(:,mmmm),[ones(size(train_covar,1),1),train_covar]); % save the residuals
        end
        [~,~,rx] = regress(train_behav,[ones(size(train_covar,1),1),train_covar]); % save the residuals
        [r_mat,p_mat] = corr(ry,rx);
    end
    
    r_mat = reshape(r_mat,no_node,no_node);
    p_mat = reshape(p_mat,no_node,no_node);
    
    % set threshold and define masks
    
    pos_mask = zeros(no_node,no_node);
    neg_mask = zeros(no_node,no_node);
    
    pos_edges = find(r_mat > 0 & p_mat < thresh);
    neg_edges = find(r_mat < 0 & p_mat < thresh);
    leng_posedge(leftout) = length(pos_edges);
    if leftout ==1
        common_edges = pos_edges;
    else
        common_edges = intersect(common_edges,pos_edges);
    end
    
    leng_negedge(leftout) = length(neg_edges);
    if leftout ==1
        common_edges_neg = neg_edges;
    else
        common_edges_neg = intersect(common_edges_neg,neg_edges);
    end
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    
    % get sum of all edges in TRAIN subs (divide by 2 to control for the
    % fact that matrices are symmetric)
    
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:size(train_sumpos);
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask,'omitnan'),'omitnan')/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask,'omitnan'),'omitnan')/2;
    end
    
    % build model on TRAIN subs
    
    fit_pos = polyfit(train_sumpos, train_behav,1);
    fit_neg = polyfit(train_sumneg, train_behav,1);
    fit_all = regress(train_behav,[ones(size(train_sumneg,1),1),train_sumpos,train_sumneg]);
    
    % run model on TEST sub
    
    test_sumpos = sum(sum(test_mat.*pos_mask,'omitnan'),'omitnan')/2;
    test_sumneg = sum(sum(test_mat.*neg_mask,'omitnan'),'omitnan')/2;
    
    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);
    behav_pred_all(leftout) = fit_all(1) + fit_all(2)*test_sumpos + fit_all(3)*test_sumneg;
    
end

[R_pos, P_pos] = corr(behav_pred_pos,all_behav);
[R_neg, P_neg] = corr(behav_pred_neg,all_behav);
[R_all, P_all] = corr(behav_pred_all,all_behav);
MSE_pos = sum((behav_pred_pos - all_behav).^2) / (no_sub - length(fit_pos)-1); %compute mean squared error (MSE)
MSE_neg = sum((behav_pred_neg - all_behav).^2) / (no_sub - length(fit_neg)-1);
MSE_all = sum((behav_pred_all - all_behav).^2) / (no_sub - length(fit_all)-1);

pre_behav_pos(:) = behav_pred_pos;
pre_behav_neg(:) = behav_pred_neg;
pre_behav_all(:) = behav_pred_all;

actual_behav(:) = all_behav;

outcome.all_behav.data = actual_behav;
outcome.R_pos = R_pos;
outcome.R_neg = R_neg;
outcome.R_all = R_all;
outcome.thresh = thresh;
outcome.pre_behav_pos =  pre_behav_pos;
outcome.pre_behav_neg =  pre_behav_neg;
outcome.pre_behav_all =  pre_behav_all;
outcome.MSE_pos = MSE_pos;
outcome.MSE_neg = MSE_neg;
outcome.MSE_all = MSE_all;
outcome.common_edges = common_edges;
outcome.leng_posedge = leng_posedge;

outcome.common_edges_neg = common_edges_neg;
outcome.leng_negedge = leng_negedge;

save([outpath filesep 'outcome_' num2str(thresh) '_' out_name '_' corr_type '_' covar_name  '.mat'],'outcome','-v7');


