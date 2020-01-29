function outcome = predfunc_Nfolds(all_mats,all_behav,thresh,Pre_Method,outpath,Nfolds,out_name,corr_type)
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

eachpart = fix(no_sub/Nfolds);
randID = randperm(no_sub);

for j = 1:Nfolds
    origID {j} = randID([(j-1)*eachpart + 1:j*eachpart])';
end

remain = mod(no_sub,Nfolds);

for j=1:remain
    origID{j} = [origID{j}; randID(Nfolds*eachpart+j)];
end


for leftout = 1:Nfolds;
    fprintf('\n Leaving out Fold # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,origID{leftout}) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3)); %from 3-dimension to 2-dimension
    
    train_behav = all_behav;
    train_behav(origID{leftout}) = [];

    test_mat = all_mats(:,:,origID{leftout});
    test_vect = reshape(test_mat,[],size(test_mat,3)); %from 3-dimension to 2-dimension
    test_behav = all_behav(origID{leftout});
    
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
        test_vect=test_vect';
%         [ms,ns] = size(test_mat);
%         test_mat = reshape(test_mat, 1,ms*ns);
        MeanValue_New = repmat(MeanValue, length(test_behav), 1);
        StandardDeviation_New = repmat(StandardDeviation, length(test_behav), 1);
        test_vect = (test_vect - MeanValue_New) ./ StandardDeviation_New;
        %test_vect = reshape(test_vect, ms,ns);
         test_vect = test_vect';
        test_mat = reshape(test_vect,size(test_mat,1),size(test_mat,2),size(test_mat,3));
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_vect=test_vect';
%         [ms,ns] = size(test_vect);
%         test_vect = reshape(test_vect, 1,ms*ns);
        MaxValue_New = repmat(MaxValue, length(test_behav), 1);
        MinValue_New = repmat(MinValue, length(test_behav), 1);
        test_vect = (test_vect - MinValue_New) ./ (MaxValue_New - MinValue_New);
        %test_vect = reshape(test_vect, ms,ns);
        test_vect = test_vect';
        test_mat = reshape(test_vect,size(test_mat,1),size(test_mat,2),size(test_mat,3));
    end
    
    % correlate all edges with behavior
    
    [r_mat,p_mat] = corr(train_vcts',train_behav,'type',corr_type); %correlate each edge with behavior
    
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
    
    train_sumpos = zeros(length(train_behav),1);
    train_sumneg = zeros(length(train_behav),1);
    
    for ss = 1:size(train_sumpos);
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask,'omitnan'),'omitnan')/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask,'omitnan'),'omitnan')/2;
    end
    
    % build model on TRAIN subs
    
    fit_pos = polyfit(train_sumpos, train_behav,1);
    fit_neg = polyfit(train_sumneg, train_behav,1);
    fit_all = regress(train_behav,[ones(size(train_sumneg,1),1),train_sumpos,train_sumneg]);
    
    % run model on TEST sub
    pred_pos = [];
    pred_neg = [];
    pred_all = [];
    for sn = 1:length(test_behav)
        test_sumpos = sum(sum(test_mat(:,:,sn).*pos_mask,'omitnan'),'omitnan')/2;
        test_sumneg = sum(sum(test_mat(:,:,sn).*neg_mask,'omitnan'),'omitnan')/2;
        
        pred_pos(sn) = fit_pos(1)*test_sumpos + fit_pos(2);
        pred_neg(sn) = fit_neg(1)*test_sumneg + fit_neg(2);
        pred_all(sn) = fit_all(1) + fit_all(2)*test_sumpos + fit_all(3)*test_sumneg;
    end
    
    behav_pred_pos{leftout} = pred_pos;
    behav_pred_neg{leftout} = pred_neg;
    behav_pred_all{leftout} = pred_all;
    
end
actual_behav = all_behav;
outcome.all_behav.data = actual_behav;
outcome.thresh = thresh;
outcome.pre_behav_pos =  behav_pred_pos;
outcome.pre_behav_neg =  behav_pred_neg;
outcome.pre_behav_all =  behav_pred_all;
outcome.origID = origID;
outcome.common_edges = common_edges;
outcome.leng_posedge = leng_posedge;

outcome.common_edges_neg = common_edges_neg;
outcome.leng_negedge = leng_negedge;

save([outpath,'\outcome_' num2str(thresh) '_' out_name '_' corr_type '.mat'],'outcome','-v7');
