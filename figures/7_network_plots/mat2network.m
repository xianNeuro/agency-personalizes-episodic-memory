function [G,diG] = mat2network(similmat,threshold,nodenames)
% create a network (directed/undirected gragh) from a pairwise similarity matrix 
% similmat = similarity matrix (n-by-n)
% threshold = edge weight threshold (r same as/lower than this value will
% be excluded)
% nodenames = cell containing node name strings (number of elements =
% number of nodes)
% 12/18/18 hongmi lee

% number of elements
numseg = size(similmat,1);

% lower triangle index
ltriidx = find(~triu(ones(numseg,numseg)));

% node indices
[idx1,idx2] = meshgrid(1:numseg,1:numseg);

idx1 = idx1(ltriidx);
idx2 = idx2(ltriidx);

% edge weights
weights = similmat(ltriidx);

% threshold edge weights
cutidx = find(weights <= threshold);
idx1(cutidx) = [];
idx2(cutidx) = [];
weights(cutidx) = [];

% create graph
G = graph(idx1,idx2,weights,nodenames);
diG = digraph(idx1,idx2,weights,nodenames); % directed

end

