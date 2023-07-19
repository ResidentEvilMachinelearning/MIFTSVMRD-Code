function [fp, fn]= ftsvmpreddists(ftsvm_struct,Testdata)
% Function:  testing iftsvm on test data
% Input:
% Iftsvm_struct         - the trained  ftsvm model
% Testdata             - test data
% Testlabel            - test label
%
% Output:
% acc                    - accuracy
% outclass               - predict label
%% check the number of arguments
if ( nargin>3||nargin<2) % check correct number of
    help ftsvmclass
else
    %% get vp and vn from the train model
    vp = ftsvm_struct.vp;
    vn = ftsvm_struct.vn;
    %% compute fp and fn
    switch ftsvm_struct.Parameter.ker
        case 'linear'
            fp = (Testdata*vp(1:(length(vp)-1))+vp(length(vp)))./norm(vp(1:(length(vp)-1)));
            fn = (Testdata*vn(1:(length(vn)-1))+vn(length(vn)))./norm(vn(1:(length(vn)-1)));
        case 'knn'
            fp = (Testdata*vp(1:(length(vp)-1))+vp(length(vp)))./norm(vp(1:(length(vp)-1)));
            fn = (Testdata*vn(1:(length(vn)-1))+vn(length(vn)))./norm(vn(1:(length(vn)-1)));
        case 'rbf'
            X = ftsvm_struct.X;
            kfun = ftsvm_struct.KernelFunction;
            kfunargs = ftsvm_struct.KernelFunctionArgs;
            K = feval(kfun,Testdata,X,kfunargs{:});
            fp = (K*vp(1:(length(vp)-1))+vp(length(vp)))./norm(vp(1:(length(vp)-1)));
            fn = (K*vn(1:(length(vn)-1))+vn(length(vn)))./norm(vn(1:(length(vn)-1)));
    end
    % normalization of fp and fn
    temp = sqrt(fp.^2 + fn.^2);
    fp = fp./temp;
    fn = fn./temp;
end