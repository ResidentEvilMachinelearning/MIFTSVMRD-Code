function  [alpha ,v,iter] = L1CD(H,G,cm,cp,eps,max_iter)
% Function:  dual  coordinate descent  for the ftsvm
% Input:
% H,G,cm,cp,eps,max_iter
%
% Output:
% alpha ,v,iter
%


if ( nargin>6||nargin<4) % check correct number of arguments
    help L1CD
else
    [~,columnH]=size(H);
    rowG=size(G,1);
    
    if (nargin<5)
        eps=0.001;
    end
    if (nargin<6)
        max_iter=200;
    end
    
    
    E=eye(columnH);
    E(columnH,columnH)=0;
    %% compute Q_bar and Qii
    Q_bar=(H'*H+cm*E)\G';
    for  i=1:rowG
        Q(i)=G(i,:)*Q_bar(:,i);
    end
        
    X_new = 1:rowG;
    X_old = 1:rowG;
    
    alpha  = zeros(rowG,1); 
    alphaold = zeros(rowG,1);
    v = zeros(columnH,1); 
    
    PGmax_old = inf;       %M_bar
    PGmin_old = -inf;      %m_bar
    
    iter = 1;    
    while iter<max_iter
        %1 While
        PGmax_new = -inf;   %M
        PGmin_new = +inf;   %m
        R = length(X_old);
        X_old = X_old(randperm(R));
        %2 for
        for  j = 1:R
            i = X_old(j);
            pg = -G(i,:)*v-1;  % f(alpha)
            PG = 0;               
            if alpha(i) == 0    % PG = min(0,f(alpha))
                if pg>PGmax_old
                    X_new(X_new==i) = []; % PG = 0
                    continue;
                elseif  pg<0
                    PG = pg;            % PG = f(alpha)
                end
            elseif alpha(i)==cp(i) % alpha = c2, PG = max(0,f(alpha)
                if pg<PGmin_old
                    X_new(X_new==i) = []; % PG = 0
                    continue;
                elseif  pg>0
                    PG = pg;            
                end
            else  % 0 <= alpha <= c2
                PG = pg;
            end
            PGmax_new = max(PGmax_new,PG);
            PGmin_new = min(PGmin_new,PG);
            if abs(PG)> 1.0e-12
                alphaold(i,1) = alpha(i);
                alpha(i,1) = min(max(alpha(i)-PG/Q(i),0.0),cp(i));
                v = v-Q_bar(:,i)*(alpha(i,1)-alphaold(i,1));
            end
        end
        %     M(iter+1)=PGmax_new;
        %     N(iter+1)=PGmin_new;
        
        X_old = X_new;
        iter = iter+1; 
        %3
        if  PGmax_new-PGmin_new<=eps
            if length(X_old)==rowG
                break;
            else
                X_old = 1:rowG;  X_new = 1:rowG;
                PGmax_old = inf;   PGmin_old = -inf;
            end
        end
        
        %4 
        if  PGmax_new<=0
            PGmax_old = inf;
        else
            PGmin_old = PGmax_new;
        end
        %5 
        if  PGmin_old>=0
            PGmin_old = -inf;
        else
            PGmin_old = PGmin_new;
        end
        
    end
%     fprintf('convergent iteration times     : %d\n',iter);
end
end
