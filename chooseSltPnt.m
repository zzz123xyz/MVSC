function [SltPnt, SltPntInd] = chooseSltPnt(X, nbSltPnt)
%%
% Li, Y., Nie, F., Huang, H., & Huang, J. (2015, January). 
% Large-Scale Multi-View Spectral Clustering via Bipartite Graph. In AAAI (pp. 2750-2756).
% by Lance Liu 27/05/16

%input
% X: R^{n \times d} n samples d dimensions
% nbSltPnt: number of salient points

%output
% SltPnt: salient points
% SltPntInd: index of salient points

%%
[~, C] = kmeans(X, nbSltPnt);
SltPntInd = zeros(0,nbSltPnt);

[score, tmp1] = sort(pdist2(X,C));
SltPntInd = tmp1(1,:);

SltPnt = X(SltPntInd, :);