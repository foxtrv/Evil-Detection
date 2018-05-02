function [ output_args ] = do_svm( config_file )
%% Function that runs the Naive Bayes classifier on histograms of
%% vector-quantized image regions. Based on the paper:
%% 
%% Visual categorization with bags of keypoints
%% Chris Dance, Jutta Willamowski, Lixin Fan, Cedric Bray, Gabriela Csurka
%% ECCV International Workshop on Statistical Learning in Computer Vision, Prague, 2004. 
%% http://www.xrce.xerox.com/Publications/Attachments/2004-010/2004_010.pdf
  
%% Note that this only trains a model. It does not evaluate any test
%% images. Use do_naive_bayes_evaluation for that.  
  
%% Before running this, you must have run:
%%    do_random_indices - to generate random_indices.mat file
%%    do_preprocessing - to get the images that the operator will run on  
%%    do_interest_op  - to get extract interest points (x,y,scale) from each image
%%    do_representation - to get appearance descriptors of the regions  
%%    do_vq - vector quantize appearance of the regions
  
%% R.Fergus (fergus@csail.mit.edu) 03/10/05.  
 
    
%% Evaluate global configu
%% Evaluate global configuration file
eval(config_file);

%% ensure models subdir is present
[s,m1,m2]=mkdir(RUN_DIR,Global.Model_Dir_Name);

%% get all file names of training image interest point files

%% get +ve interest point file names
pos_ip_file_names = [];
pos_sets = find(Categories.Labels==1);
for a=1:length(pos_sets)
    pos_ip_file_names =  [pos_ip_file_names , genFileNames({Global.Interest_Dir_Name},Categories.Train_Frames{pos_sets(a)},RUN_DIR,Global.Interest_File_Name,'.mat',Global.Num_Zeros)];
end

%% get -ve interest point file names
neg_ip_file_names = [];
neg_sets = find(Categories.Labels==0);
for a=1:length(neg_sets)
    neg_ip_file_names =  [neg_ip_file_names , genFileNames({Global.Interest_Dir_Name},Categories.Train_Frames{neg_sets(a)},RUN_DIR,Global.Interest_File_Name,'.mat',Global.Num_Zeros)];
end

%% Create matrix to hold word histograms from +ve images
X_fg = zeros(VQ.Codebook_Size,length(pos_ip_file_names));

%% load up all interest_point files which should have the histogram
%% variable already computed (performed by do_vq routine).
for a=1:length(pos_ip_file_names)
    %% load file
    load(pos_ip_file_names{a});
    %% store histogram
    X_fg(:,a) = histg';    
end 


%% Create matrix to hold word histograms from -ve images
X_bg = zeros(VQ.Codebook_Size,length(neg_ip_file_names));

%% load up all interest_point files which should have the histogram
%% variable already computed (performed by do_vq routine).
for a=1:length(neg_ip_file_names)
    %% load file
    load(neg_ip_file_names{a});
    %% store histogram
    X_bg(:,a) = histg';    
end 
xTrain = [X_fg';X_bg'];
labels = [ones(1, length(pos_ip_file_names)), zeros(1, length(neg_ip_file_names))];

svmmodel = fitcsvm(xTrain, labels,'KernelFunction','RBF','KernelScale','auto',...
    'ClassNames',[1, 0]);
%%% Compute ROC and RPC on training data
[predictions, score1]= predict(svmmodel, xTrain);

%%
vals = score1(:,1);%-score1(:,2);
%% positive 
Pw_pos = (1 + sum(X_fg,2)) / (VQ.Codebook_Size + sum(sum(X_fg)));

%% positive 
Pw_neg = (1 + sum(X_bg,2)) / (VQ.Codebook_Size + sum(sum(X_bg)));

[roc_curve_train,roc_op_train,roc_area_train,roc_threshold_train] = roc([vals';labels]');
fprintf('Training: Area under ROC curve = %f; Optimal threshold = %f\n', roc_area_train, roc_threshold_train);

[rpc_curve_train,rpc_ap_train,rpc_area_train,rpc_threshold_train] = recall_precision_curve([vals';labels]',length(pos_ip_file_names));
fprintf('Training: Area under RPC curve = %f\n', rpc_area_train);

[fname,model_ind] = get_new_model_name([RUN_DIR,'/',Global.Model_Dir_Name],Global.Num_Zeros);


save(fname,'Pw_pos','Pw_neg','roc_curve_train','roc_op_train','roc_area_train','roc_threshold_train','rpc_curve_train','rpc_ap_train','rpc_area_train','rpc_threshold_train','svmmodel');

config_fname = which(config_file);
copyfile(config_fname,[RUN_DIR,'/',Global.Model_Dir_Name,'/',Global.Config_File_Name,prefZeros(model_ind,Global.Num_Zeros),'.m']);


end

