function [sp,sn,sz,XPnoise,XNnoise,XZnoise,time]=fuzzy3(Xp,Xn,Xz,Parameter)
% Function:  compute fuzzy membership - 3 classes
% Input:      
% Xp                        -  the positive samples
% Xn                        -  the  negative samples
% Xz                        -  the  zero samples (other classes)
% Parameter         -  the parameters 
%
% Output:    
% sp                         - the fuzzy mebership vlaue for Xp
% sn                         - the fuzzy mebership vlaue for Xn
% sz                         - the fuzzy mebership vlaue for Xz
%

%% check the number of arguments
if ( nargin>4||nargin<4) % check correct number of arguments
    help Gbbftsvm
else
    if (nargin<4) 
        Parameter.ker='linear'; 
    end

     %% setting parameters
    eplison=1e-10;
    rxp = size(Xp,1);
    rxn = size(Xn,1);
    rxz = size(Xz,1);
    %% setting kernel
    switch Parameter.ker
        case 'linear'
            kfun = @linear_kernel;kfunargs ={};
        case 'knn'
            kfun = @linear_kernel;kfunargs ={};
        case 'rbf'
            p1=Parameter.p1;
            kfun = @rbf_kernel;kfunargs = {p1};
    end
    %  ||(xi+)-cen+||^2
    switch lower(Parameter.ker)
        case 'linear'
            tic;
            Xp_cen=mean(Xp);
            Xn_cen=mean(Xn);
            Xz_cen=mean(Xz);


            radiusxp=sum((repmat(Xp_cen,rxp,1)-Xp).^2,2);
            radiusmaxxp=max(radiusxp);
            radiusxpxn=sum((repmat(Xn_cen,rxp,1)-Xp).^2,2);
            radiusxpxz=sum((repmat(Xz_cen,rxp,1)-Xp).^2,2);
            radiusxpxnxz = min(radiusxpxn, radiusxpxz);  

            
            radiusxn=sum((repmat(Xn_cen,rxn,1)-Xn).^2,2);
            radiusmaxxn=max(radiusxn);
            radiusxnxp=sum((repmat(Xp_cen,rxn,1)-Xn).^2,2);
            radiusxnxz=sum((repmat(Xz_cen,rxn,1)-Xn).^2,2);
            radiusxnxpxz = min(radiusxnxp, radiusxnxz);

            
            
            radiusxz=sum((repmat(Xz_cen,rxz,1)-Xz).^2,2);
            radiusmaxxz=max(radiusxz);
            radiusxzxp=sum((repmat(Xp_cen,rxz,1)-Xz).^2,2); 
            radiusxzxn=sum((repmat(Xn_cen,rxz,1)-Xz).^2,2);
            radiusxzxpxn = min(radiusxzxp, radiusxzxn);

            
            Ap=zeros(rxp,1);
            XPnoise=find(radiusxp>=radiusxpxnxz);
            XPnormal=find(radiusxp<radiusxpxnxz);
            Ap(XPnormal,1)=(1-abs(radiusxp(XPnormal,1))./(radiusmaxxp+eplison));
            Ap(XPnoise,1)=(1-abs(radiusxp(XPnoise,1))./(radiusmaxxp+eplison));
            
            An=zeros(rxn,1);
            XNnoise=find(radiusxn>=radiusxnxpxz);
            XNnormal=find(radiusxn<radiusxnxpxz);
            An(XNnormal,1)=(1-abs(radiusxn(XNnormal,1))./(radiusmaxxn+eplison));
            An(XNnoise,1)=(1-abs(radiusxn(XNnoise,1))./(radiusmaxxn+eplison));
            

            Az=zeros(rxz,1);
            XZnoise=find(radiusxz>=radiusxzxpxn);
            XZnormal=find(radiusxz<radiusxzxpxn);
            Az(XZnormal,1)=(1-abs(radiusxz(XZnormal,1))./(radiusmaxxz+eplison));
            Az(XZnoise,1)=(1-abs(radiusxz(XZnoise,1))./(radiusmaxxz+eplison));
            
            Bp=(1-Ap).*-1;
            Bn=(1-An).*-1;
            Bz=(1-Az).*-1;
            
            sp=zeros(0);
            if Bp==Ap
               sp=Ap;
            elseif Ap<=Bp
               sp=0;
            else
               sp=(1-Bp)./(2-Ap-Bp);
            end
            
            sn=zeros(0);
            if Bn==An
               sn=An;
            elseif An<=Bn
               sp=0;
            else
               sn=(1-Bn)./(2-An-Bn);
            end
            
            sz=zeros(0);
            if Bz==Az
               sz=Az
            elseif Az<=Bz
               sz=0;
            else
               sz=(1-Bz)./(2-Az-Bz);
            end
            
            
            time=toc;
            
        case 'rbf'
            tic;
            kerxp1 = feval(kfun,Xp,Xp,kfunargs{:});
            kerxp2 = feval(kfun,Xn,Xn,kfunargs{:});
            kerxp3 = feval(kfun,Xz,Xz,kfunargs{:});
            kerxp4 = feval(kfun,Xn,Xz,kfunargs{:});
            kerxp5 = feval(kfun,Xp,Xz,kfunargs{:});
            kerxp6 = feval(kfun,Xp,Xn,kfunargs{:});
       

            radiusxp=1-2*mean(kerxp1,2)+mean(mean(kerxp1));
            radiusxpxn1=1-2*mean(kerxp6,2)+mean(mean(kerxp2));
            radiusxpxz2=1-2*mean(kerxp5,2)+mean(mean(kerxp3));
            radiusxpxn=radiusxpxn1+radiusxpxz2;
            radiusmaxxp=max(radiusxp);
            
            radiusxn=1-2*mean(kerxp2,2)+mean(mean(kerxp2));  
            radiusxnxp1=1-2*transpose(mean(kerxp6,1))+mean(mean(kerxp1));
            radiusxnxz2=1-2*transpose(mean(kerxp4,1))+mean(mean(kerxp3));
            radiusxnxp=radiusxnxp1+radiusxnxz2;
            radiusmaxxn=max(radiusxn);

            radiusxz=1-2*mean(kerxp3,2)+mean(mean(kerxp3)); 
            radiusxzxn1=1-2*mean(kerxp4,2)+mean(mean(kerxp2));
            radiusxzxp2=1-2*mean(kerxp5,2)+mean(mean(kerxp1));
            radiusxzxn=radiusxzxn1+radiusxzxp2;
            radiusmaxxz=max(radiusxz);



            Ap=zeros(rxp,1);
            XPnoise=find(radiusxp>=radiusxpxn);
            XPnormal=find(radiusxp<radiusxpxn);
            Ap(XPnormal,1)=(1-abs(radiusxp(XPnormal,1))./(radiusmaxxp+eplison));
            Ap(XPnoise,1)=(1-abs(radiusxp(XPnoise,1))./(radiusmaxxp+eplison));
            
            An=zeros(rxn,1);
            XNnoise=find(radiusxn>=radiusxnxp);
            XNnormal=find(radiusxn<radiusxnxp);
            An(XNnormal,1)=(1-abs(radiusxn(XNnormal,1))./(radiusmaxxn+eplison));
            An(XNnoise,1)=(1-abs(radiusxn(XNnoise,1))./(radiusmaxxn+eplison));

            Az=zeros(rxz,1);
            XZnoise=find(radiusxz>=radiusxzxn);
            XZnormal=find(radiusxz<radiusxzxn);
            Az(XZnormal,1)=(1-abs(radiusxz(XZnormal,1))./(radiusmaxxz+eplison));
            Az(XZnoise,1)=(1-abs(radiusxz(XZnoise,1))./(radiusmaxxz+eplison));


            Bp=(1-Ap).*0.1;
            Bn=(1-An).*0.1;
            Bz=(1-Az).*0.1;
            
            sp=zeros(0);
            if Bp==Ap
               sp=Ap
            elseif Ap<=Bp
               sp=0
            else
               sp=(1-Bp)./(2-Ap-Bp)
            end
            

            sn=zeros(0);
            if Bn==An
               sn=An;
            elseif An<=Bn
               sp=0;
            else
               sn=(1-Bn)./(2-An-Bn);
            end
            

            sz=zeros(0);
            if Bz==Az
               sz=Az;
            elseif Az<=Bz
               sz=0;
            else
               sz=(1-Bz)./(2-Az-Bz);
            end
            
            
            time=toc;
            % fprintf('compute fuzzy function time: %.2f s\n',time);
    end
      sp=mapminmax(sp',eps,1)';
      sn=mapminmax(sn',eps,1)';
      sz=mapminmax(sz',eps,1)';
end

