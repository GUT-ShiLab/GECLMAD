import torch as th
from torch import nn,einsum
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
from get_microbe import GraphTransformer as GraphTransformer1
from get_disease import GraphTransformer as GraphTransformer2
from GAT import GAT



class DMCFMDA(nn.Module):
    def __init__(self, args):
        super(DMCFMDA, self).__init__()
        self.args = args
        self.runtime_device = th.device(getattr(args, "device", "cpu"))
        self.lin_m = nn.Linear(args.microbe, args.in_feats, bias=False)
        self.lin_d = nn.Linear(args.disease_number, args.in_feats, bias=False)

        self.gt_microbe = GraphTransformer1(self.runtime_device, args.gt_layer, args.microbe, args.microbe,
                                            args.microbe, args.gt_head, args.dropout)
        self.gt_disease = GraphTransformer2(self.runtime_device, args.gt_layer, args.disease_number, args.disease_number,
                                            args.disease_number, args.gt_head, args.dropout)

        self.trans_gat_microbe = GAT(args.microbe, [128, 64], args.out_feats, num_heads=10, dropout=args.dropout)
        self.trans_gat_disease = GAT(args.disease_number, [128, 64], args.out_feats, num_heads=10,
                                          dropout=args.dropout)
        self.gat_md = GAT(args.in_feats, [128, 64], args.out_feats, num_heads=10, dropout=args.dropout)

        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(args.dropout)
        in_feat = 2 * args.out_feats
        for idx, out_feat in enumerate(args.mlp):
            if idx == 0:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('elu', nn.ELU())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat
            else:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('sigmoid', nn.Sigmoid())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat

    def forward(self, mm_graph, dd_graph, md_graph, microe, disease, samples, epoch,
                Issne=False, mm_edge_index=None, dd_edge_index=None, md_edge_index=None):
        mm_graph = mm_graph.to(self.runtime_device)
        dd_graph = dd_graph.to(self.runtime_device)
        md_graph = md_graph.to(self.runtime_device)
        microe = microe.to(self.runtime_device)
        disease = disease.to(self.runtime_device)

        mi_sim = self.gt_microbe(mm_graph, microe)    #gt
        di_sim = self.gt_disease(dd_graph, disease)   #gt

        # GraphTransformer uses DGL graphs, while the repository's GAT
        # implementation uses torch-geometric edge-index tensors.  Keep both
        # representations explicit for the fixed-fold runner.
        mm_gat_graph = mm_graph if mm_edge_index is None else mm_edge_index.to(self.runtime_device)
        dd_gat_graph = dd_graph if dd_edge_index is None else dd_edge_index.to(self.runtime_device)
        md_gat_graph = md_graph if md_edge_index is None else md_edge_index.to(self.runtime_device)

        emb_m = self.trans_gat_microbe(mm_gat_graph, mi_sim, microe)
        emb_d = self.trans_gat_disease(dd_gat_graph, di_sim, disease)

        combined_features = th.cat((self.lin_m(microe), self.lin_d(disease)), dim=0)
        emb_md = self.gat_md(md_gat_graph, combined_features, combined_features)

        emb_mm_ass = emb_md[:self.args.microbe, :]
        emb_dd_ass = emb_md[self.args.microbe:, :]

        emb_mm = emb_m + emb_mm_ass
        emb_dd = emb_d + emb_dd_ass

        emb = th.cat((emb_mm[samples[:, 0]], emb_dd[samples[:, 1]]), dim=1)
        result = self.mlp(emb)
        return result, emb_m, emb_mm_ass, emb_d, emb_dd_ass
