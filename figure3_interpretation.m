
input_dir = 'C:\Users\DeyingLi\Desktop\KRR_Yeo\data_dyli\features_FGE.txt';
data = readmatrix(input_dir);
data = transpose(data);

results =load('C:\Users\DeyingLi\Desktop\KRR_Yeo\KRR_FGE_results\KRR_FGE_PCA\seed_1\results\final_result_KRR_FGE_PCA.mat');
load('C:\Users\DeyingLi\Desktop\KRR_Yeo\KRR_FGE_results\KRR_FGE_PCA\seed_1\results\no_relative_10_fold_sub_list.mat')

% load features
feat = data'; 

% pre allocate space for cov_mat
cov_mat = zeros(size(sub_fold,1), size(feat,1), ...
    size(results.y_pred_train{1},2));

% calculate feature importance for each fold
for i = 1:size(sub_fold,1)
    % find train fold idx
    train = ~sub_fold(i).fold_index;
    fold_name = strcat('fold_', num2str(i));
    % load features and normalize
    feat_train = feat(:,train);
    feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);
    % load predictions
    y_pred = results.y_pred_train{i};

    % compute covariance
    for b = 1:size(y_pred,2)
        cov_mat(i,:,b) = bsxfun(@minus,feat_train_norm,mean(feat_train_norm,2)) * ...
            bsxfun(@minus,y_pred(:,b),mean(y_pred(:,b))) / (size(feat_train_norm,2));
    end
end

cov_mat_mean = squeeze(mean(cov_mat,1));