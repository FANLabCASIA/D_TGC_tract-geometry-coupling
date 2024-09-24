%{
Ref: Decoupling of brain function from structure reveals regional behavioral specialization in humans
Code ref: github.com/gpreti/GSP_StructuralDecouplingIndex
Authors: Maria Giulia Preti & Dimitri Van De Ville 
%}

%% 0. config
age_name = 'BL';
surface_name = 'white';
hemi = 'l';
disp(strcat(age_name, ':', surface_name, hemi, 'h'));

%% 1. Loading required data
% X_RS=load('/n01dat01/dyli/multi/results_data/BL_white_fingerprint_eachsub_v3.mat');
X_RS=load('/n01dat01/dyli/multi/results_data/HCP_white_fingerprint_eachsub_norm.mat');
X_RS=X_RS.X_RS;

%% zscore fMRI timecourses
% zX_RS=zscore(X_RS,0,2);
zX_RS = X_RS;

%%
% Number of regions
n_ROI = 200;
% number of subjects
nsubjs_RS=size(zX_RS,3);

%% 2. Laplacian Decomposition results
% U = load(strcat('/n01dat01/dyli/multi/support_code/BrainEigenmodes/data/template_eigenmodes/',age_name,'_fsLR_32k_',surface_name,'-',hemi,'h_emode_200_forCDindex.txt'));
% LambdaL = load(strcat('/n01dat01/dyli/multi/support_code/BrainEigenmodes/data/template_eigenmodes/',age_name,'_fsLR_32k_',surface_name,'-',hemi,'h_eval_200.txt'));
U = load(strcat('/n01dat01/dyli/multi/support_code/BrainEigenmodes/data/template_eigenmodes/fsLR_32k_',surface_name,'-',hemi,'h_emode_200_forCDindex.txt'));
LambdaL = load(strcat('/n01dat01/dyli/multi/support_code/BrainEigenmodes/data/template_eigenmodes/fsLR_32k_',surface_name,'-',hemi,'h_eval_200.txt'));

%% 3. Average energy spectral density of resting-state functional data projected on the structural harmonics
clear X_hat_L
for s=1:nsubjs_RS
    X_hat_L(:,:,s)=U'*zX_RS(:,:,s);
end
pow=abs(X_hat_L).^2;
PSD=squeeze(mean(pow,2));
save(strcat('/n01dat01/dyli/multi/results_data/decoupling_index/',age_name,'_fsLR_32k_',surface_name,'-',hemi,'h_PSD_200_forCDindex.txt'), 'PSD', '-ascii')

avg=mean(PSD')';
stdPSD=std(PSD')';
upper1=avg+stdPSD;
lower1=avg-stdPSD;
idx = max(PSD')>0 & min(PSD')>0 & mean(PSD')>0;                       

%% 4. compute cut-off frequency
mPSD=mean(PSD,2);
AUCTOT=trapz(mPSD(1:200)); %total area under the curve

i=0;
AUC=0;
while AUC<AUCTOT/2
    AUC=trapz(mPSD(1:i));
    i=i+1;
end
NN=i-1; %CUTOFF FREQUENCY C : number of low frequency eigenvalues to consider in order to have the same energy as the high freq ones
NNL=200-NN; 
disp(NN);
disp(NNL);

%% 5. split structural harmonics in high/low frequency

M=fliplr(U); %Laplacian eigenvectors flipped in order (high frequencies first)

Vlow=zeros(size(M));
Vhigh=zeros(size(M));
Vhigh(:,1:NNL)=M(:,1:NNL);%high frequencies= decoupled 
Vlow(:,end-NN+1:end)=M(:,end-NN+1:end);%low frequencies = coupled 
