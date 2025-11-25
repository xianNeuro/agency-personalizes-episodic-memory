

 similaritymat = importdata('sub2_SemanticMat_37nodes.csv');
% similaritymat = importdata('sub2_CausalityMat_37Events.csv');
% similaritymat = importdata('sub4_SemanticMat_20nodes.csv');
% similaritymat = importdata('sub4_CausalityMat_20Events.csv');

%options for color map (sub2 use 4-orange, sub4 use 2-blue)
gcolormap = [
    253,191,111;
    255,127,0;
    166,206,227;
    31,120,180;
    178,223,138;
    51,160,44;
    251,154,153;
    227,26,28;
    202,178,214;
    106,61,154]/255;

layout = 'force';
nodenames=string(1:length(similaritymat));
[G diG]= mat2network(similaritymat,0.1,nodenames);

% node size (centrality weighted)
degree = centrality(G,'degree','Importance',G.Edges.Weight);
degree_norm = zscore(degree./sum(degree));
nodesize = 1+10*(degree_norm-min(degree_norm))/(max(degree_norm)-min(degree_norm));

% line width (similarity weighted)
LWidths = 2*G.Edges.Weight/max(G.Edges.Weight);

% draw networks (%%%%% edge color = EdgeCdata)
h = plot(G,'LineWidth',LWidths,'Layout',layout,'Marker','o','NodeLabel',{},...
    'MarkerSize',nodesize,'EdgeCData',G.Edges.Weight,'NodeColor',gcolormap(4,:));%'NodeColor',moviecolormap(i,:)

set(gca,'box','off');
ax1 = gca; ax1.YAxis.Visible = 'off'; ax1.XAxis.Visible = 'off';