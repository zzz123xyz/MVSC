function [Y, C, obj_value, SltPnt] = MVSC(data, nbclusters, nbSltPnt, k, gamma...
, method, param)
%%
% Li, Y., Nie, F., Huang, H., & Huang, J. (2015, January). 
% Large-Scale Multi-View Spectral Clustering via Bipartite Graph. In AAAI (pp. 2750-2756).

% by Lance Liu 20/05/16

%input
% data: for k views of features, k row cells with R^{d \times n} in each
% cell
% nbclusters: number of clusters
% nbSltPnt: number of salient points
% k: k nearest neighbors
% gamma: parameter

%output
% ???
% 

%%
niters = 10;

V = numel(data);

[dim_V_ind1, dim_V_ind2] = DataSplitIndex(data);


[X, n, nfeat] = DataConcatenate(data);

[SltPnt, SltPntInd] = chooseSltPnt(X', nbSltPnt);
RestPntInd = setdiff([1:n],SltPntInd);


a = repmat(1/nbclusters, [1,V]);

%remove SltPnt from X

for v = 1:V
    
    RestPnt = data{v}(:,RestPntInd)';
    PairDist = pdist2(RestPnt,SltPnt(:,dim_V_ind1(v): dim_V_ind2(v)));
    [score, ind] = sort(PairDist,2);
    ind = ind(:,1:k);   
    
%*****
%make a Indicator Mask to record j \in \theta_i
    IndMask = zeros(n - nbSltPnt, nbSltPnt);
    for i = 1:n - nbSltPnt
         IndMask(i, ind(i,:)) = 1;
    end
    
    
    Kernel = exp(-(PairDist).^2 ./ (2*param^2));
    Kernel = Kernel.*IndMask;
    
    SumSltKnl = repmat(sum(Kernel, 2),[1,nbSltPnt]);
    Z{v} = Kernel ./ SumSltKnl;
    Dc{v} = diag(sum(Z{v},1)+eps);
    Dr{v} = diag(sum(Z{v},2));
    D{v} = blkdiag(Dr{v},Dc{v});
    
    tmp1 = zeros(n);
    tmp1(1:n-nbSltPnt,n-nbSltPnt+1:end) = Z{v};
    tmp1(n-nbSltPnt+1:end,1:n-nbSltPnt) = Z{v}';
    W{v} = tmp1;
    
    L{v} = eye(n) - (D{v}^-0.5) * W{v} * (D{v}^-0.5);
end

for t = 1:niters
    L_sum = zeros(n, n);
    Z_hat = zeros(n - nbSltPnt, nbSltPnt);
    
    for v = 1:V
        Z_hat = Z_hat + a(v)^gamma*Z{v}*(Dc{v})^(-0.5);
    end 
    
    % compute G according to (14)
    [Gx_a, S, Gu_a] = svd(Z_hat, 'econ');
    Gx = Gx_a(:,1:nbclusters);
    Gu = Gu_a(:,1:nbclusters);
    G = [Gx', Gu']';
    
    for v = 1:V
        h(v) = trace(G'*L{v}*G); %*** h(v) = trace(G'*L{v}*G); dim of G mismatch L
    end
    
    % compute a(v) according to (10)
    tmp1 = (gamma .* h).^(1/(1-gamma)) ;
    a = tmp1 ./ sum(tmp1);
    
    [Y, C] = kmeans(G, nbclusters);
    
    
    % compute the value of objective function (5)
    for v = 1:V
        L_sum = L_sum + a(v)^gamma*L{v};
    end
    obj_value(t) = trace(G'*L_sum*G);  %obj_value(t) = trace(G'*L_sum*G);
        
end









