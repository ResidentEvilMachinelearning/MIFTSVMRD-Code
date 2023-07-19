function  [ftsvm_struct] = ftsvmtrain(Traindata,Trainlabel,Parameter)
% Function:  train M-IFTSVM-RD
% Input:
% Traindata         -  the train data where the feature are stored
% Trainlabel        -  the  lable of train data
% Parameter         -  the parameters for ftsvm
%
% Output:
% ftsvm_struct      -  iftsvm model
%


%% check correct number of arguments
if ( nargin>3||nargin<3)
    help  ftsvmtrain
end
st1 = cputime;
%% get parameters
ker=Parameter.ker;
CC=Parameter.CC;
CR=Parameter.CR;

%% seperate data into 3 groups, with lable +1, -1 and 0
Xp = Traindata(Trainlabel ==  1,:);    % class +1
Xn = Traindata(Trainlabel == -1,:);    % class +1
Xz = Traindata(Trainlabel ==  0,:);    % class +1
X = [Xp;Xn;Xz];
% get lenth of each group
lp=sum(Trainlabel ==  1);
ln=sum(Trainlabel == -1);
lz=sum(Trainlabel ==  0);

%% compute fuzzy membership, 3 classese
[sp,sn,sz,NXpv,NXnv,NXzv]=fuzzy(Xp,Xn,Xz,Parameter);

%% Optimizing ...
% kernel function
switch ker
    case 'linear'
        kfun = @linear_kernel;kfunargs ={};
        Kpx=Xp;Knx=Xn;Kzx=Xz;
    case 'knn'
        kfun = @linear_kernel;kfunargs ={};
        Kpx=Xp;Knx=Xn;Kzx=Xz;
    case 'rbf'
        p1=Parameter.p1;
        kfun = @rbf_kernel;kfunargs = {p1};
        Kpx = feval(kfun,Xp,X,kfunargs{:});%K(X+,X)
        Knx = feval(kfun,Xn,X,kfunargs{:});%K(X-,X)
end

S=[Kpx ones(lp,1)];
R=[Knx ones(ln,1)];
Z=[Kzx ones(lz,1)];

CC1=CC*sn;
CC2=CC*sp;
CC3=CC*sz;


CC1_3 = [CC*sp; CC*sz];
CC2_3 = [CC*sn; CC*sz];


Y1 = [S; Z];
Y2 = [R; Z];


% fprintf('Optimising ...\n');
switch  Parameter.algorithm
    case  'CD'
        [alpha ,vp] =  L1CD(R,Y1,CR,CC1_3);
        [beta , vn] =  L1CD(S,Y2,CR,CC2_3);
        vn=-vn;
end
ExpendTime=cputime - st1;


ftsvm_struct.X = X;
ftsvm_struct.sp = sp;
ftsvm_struct.sn = sn;

ftsvm_struct.alpha = alpha;
ftsvm_struct.beta  = beta;
ftsvm_struct.vp = vp;
ftsvm_struct.vn = vn;

ftsvm_struct.KernelFunction = kfun;
ftsvm_struct.KernelFunctionArgs = kfunargs;
ftsvm_struct.Parameter = Parameter;
ftsvm_struct.time=ExpendTime;

ftsvm_struct.NXpv=NXpv;
ftsvm_struct.NXnv=NXnv;
ftsvm_struct.NXpv=NXzv;
ftsvm_struct.nv=length(NXpv)+length(NXnv)+length(NXzv);
end

