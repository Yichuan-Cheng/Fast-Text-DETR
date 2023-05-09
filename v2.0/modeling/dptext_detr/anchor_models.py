import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from adet.layers.anchor_deformable_transformer import DeformableTransformer_Det
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from .utils import MLP, inverse_sigmoid
import fvcore.nn.weight_init as weight_init

class DPText_DETR(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone

        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes = 1  # only text
        self.sigmoid_offset = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.epqm = cfg.MODEL.TRANSFORMER.EPQM
        self.efsa = cfg.MODEL.TRANSFORMER.EFSA
        self.ctrl_point_embed = nn.Embedding(self.num_proposals, self.d_model)

        self.transformer = DeformableTransformer_Det(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            num_proposals=self.num_proposals,
            num_ctrl_points=self.num_ctrl_points,
            epqm=self.epqm,
            efsa=self.efsa
        )
        self.ctrl_point_class = MLP(self.d_model, self.d_model, self.num_classes, 2)
        # self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.mask_embed = MLP(self.d_model, self.d_model, self.d_model, 2)
        # self.offset_embed = nn.ModuleList([MLP(self.d_model, self.d_model, 2, 3) for i in range(self.num_ctrl_points)])
        self.offset_embed = MLP(self.d_model, self.d_model, 2, 3)
        # self.offset_embed_lower = MLP(self.d_model, self.d_model, 16, 3)
        self.query_aggregation_weights = nn.Sequential(nn.Conv1d(self.d_model, self.d_model, kernel_size=9, padding='same', bias=False),
                                                        nn.ReLU(),
                                                        nn.Conv1d(self.d_model, 1, kernel_size=9, padding='same', bias=False),
                                                        nn.Sigmoid())

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model,kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            ])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.layers[-1].bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.offset_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.offset_embed.layers[-1].bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        
        self.mask_feature_proj = nn.Conv2d(
            self.d_model,
            self.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_feature_proj)
        self.ctrl_point_class = nn.ModuleList([self.ctrl_point_class for _ in range(num_pred)])
        self.offset_embed = nn.ModuleList([self.offset_embed for _ in range(num_pred)])
        # if self.epqm:
        #     self.transformer.decoder.ctrl_point_coord = self.ctrl_point_coord
        self.transformer.decoder_new.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.point_embd_norm = nn.LayerNorm(self.d_model)

        self.transformer.decoder_new.decoder_norm = self.decoder_norm
        self.transformer.decoder_new.mask_embed = self.mask_embed
        self.transformer.decoder_new.device = self.device
        self.transformer.decoder_new.mask_feature_proj = self.mask_feature_proj

        self.transformer.decoder_new.offset_embed = self.offset_embed
        self.transformer.decoder_new.point_embd_norm = self.point_embd_norm
        self.transformer.decoder_new.query_aggregation_weights = self.query_aggregation_weights
        # self.transformer.decoder.offset_embed_lower = self.offset_embed_lower

        self.to(self.device)

    def forward_prediction_heads(self, output, mask_features, query_pos, lvl):
        b, nq, n_ctrl, h = output.shape
        output = self.point_embd_norm(output.reshape(b * nq, n_ctrl, h)).reshape(b, nq, n_ctrl, h)
        decoder_output = self.decoder_norm(output.sum(-2))
        mask_features = self.mask_feature_proj(mask_features)
        # print(mask_features.shape)
        # decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.ctrl_point_class[lvl](decoder_output)

        mask_embed = self.mask_embed(decoder_output)
        
        # corrd_offset = []
        # for i in range(self.num_ctrl_points):
        #     corrd_offset.append(self.offset_embed[i](decoder_output))
        # corrd_offset = torch.stack(corrd_offset, dim=-1).transpose(-1,-2)
        # print(coord_offset.shape)
        # print(query_pos.shape, decoder_output.shape)
        corrd_offset = self.offset_embed[lvl](output)
        # corrd_offset_lower = self.offset_embed_lower(decoder_output)
        # print(corrd_offset_first.shape)
        # print(mask_embed.shape, output.shape)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
 
        b, nq, h, w = outputs_mask.shape
        # h: 1 / (h + 1) ... h / (h+1)
        # 1 * input_shape[-1] / (w + 1), w * input_shape[-1] / (w + 1)
        x, y = torch.meshgrid(torch.linspace(1 / (w + 1),w / (w + 1),w),\
                              torch.linspace(1 / (h + 1),h / (h + 1),h))
        pre_defined_anchor_grids = torch.stack((x, y), 2).transpose(0,1).float().to(self.device)
        # print(pre_defined_anchor_grids[0], 1 / (w + 1))
        # print(pre_defined_anchor_grids, h, w)
        # print(outputs_mask.shape)
        # segmentation_map = F.softmax(outputs_mask, dim=1)
        segmentation_map = outputs_mask
        outputs_anchor_weights_map = F.softmax(outputs_mask.reshape(b,nq,h*w), dim=-1).reshape(b,nq,h,w).unsqueeze(-1)

        # Weighted sum at the spatial dimension
        anchor_point = (pre_defined_anchor_grids.unsqueeze(0) * outputs_anchor_weights_map).sum(dim=(2,3)) 

        # corrds_upper = anchor_point.unsqueeze(2) + (corrd_offset_upper.reshape(b,nq,self.num_ctrl_points//2,2).sigmoid() - 0.5) * 2
        # corrds_lower = anchor_point.unsqueeze(2) + (corrd_offset_lower.reshape(b,nq,self.num_ctrl_points//2,2).sigmoid() - 0.5) * 2

        corrds = (inverse_sigmoid(anchor_point.unsqueeze(2)) + corrd_offset).sigmoid()
        # corrds_lower = sigmoid_offset(inverse_sigmoid_offset(anchor_point.unsqueeze(2)) + corrd_offset_lower.reshape(b,nq,self.num_ctrl_points//2,2))
        # subsequent_coords = [corrds_first]
        # for i in range(1,self.num_ctrl_points):
        #     subsequent_coords.append(subsequent_coords[i-1] + (self.offset_embed[i](decoder_output).unsqueeze(2).sigmoid() - 0.5) * 2)
        # print([x.shape for x in subsequent_coords])
        # corrds = torch.cat((corrds_upper, corrds_lower), dim=-2)
        # print(segmentation_map.shape)
        # print(corrds.shape)
        # print(outputs_mask.shape)
        # if outputs_mask.shape[1] > self.num_queries:
        #     outputs_mask = torch.concat([F.softmax(outputs_mask[:,:self.num_queries], dim=1), F.softmax(outputs_mask[:,self.num_queries:], dim=1)], dim=1)
        # else:
        #     outputs_mask = F.softmax(outputs_mask, dim=1)
        return outputs_class, anchor_point, corrds, segmentation_map#coord_offset.reshape(b,nq,-1,2).sigmoid()


    def forward_prediction_heads_class(self, output, lvl):
        b, nq, n_ctrl, h = output.shape
        output = self.point_embd_norm(output.reshape(b * nq, n_ctrl, h)).reshape(b, nq, n_ctrl, h)
        decoder_output = self.decoder_norm(output.sum(-2))
        outputs_class = self.ctrl_point_class[lvl](decoder_output)


        return outputs_class


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        input_shape = samples.tensor.shape
        # print(input_shape)
        features, pos = self.backbone(samples)
        
        one_fourth_feature = features[0].tensors
        features = features[1:]
        
        # for i in features:
        #     print('feature shape:', i.tensors.shape)
        

        if self.num_feature_levels == 1:
            raise NotImplementedError

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # for i in srcs:
        #     print(i.shape)
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[1]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        # Ignore one-forth faeture pos
        pos = pos[1:]
        # for i in srcs:
        #     print(i.shape)
        # n_pts, embed_dim --> n_q, n_pts, embed_dim
        # print([i.shape for i in masks])
        ctrl_point_embed = self.ctrl_point_embed.weight[:, None, :]#.repeat(self.num_proposals, 1, 1)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, \
        segmentation_maps, anchor_point = self.transformer(
            srcs, masks, pos, ctrl_point_embed, one_fourth_feature
        )
        # print(hs.shape,init_reference.shape,inter_references.shape,enc_outputs_class.shape,enc_outputs_coord_unact.shape)

        # print('refer:',inter_references.shape)
        outputs_classes = []
        outputs_coords = []
        outputs_anchors = []
        outputs_seg_mask = []

        for i, output in enumerate(hs):
            # print(output.shape)
            # outputs_class, outputs_anchor, outputs_coord, segmentation_map = self.forward_prediction_heads(output, fused_feature, query_pos[i], i)
            outputs_class = self.forward_prediction_heads_class(output, i)
            outputs_classes.append(outputs_class)
            # outputs_anchors.append(outputs_anchor)
            # outputs_coords.append(outputs_coord)
            # outputs_seg_mask.append(segmentation_map)

        # for lvl in range(hs.shape[0]):
        #     if lvl == 0:
        #         reference = init_reference
        #     else:
        #         reference = inter_references[lvl - 1]
        #     print(reference.shape)
        #     reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
        #     outputs_class = self.ctrl_point_class[lvl](hs[lvl])
        #     tmp = self.ctrl_point_coord[lvl](hs[lvl])
        #     if reference.shape[-1] == 2:
        #         if self.epqm:
        #             tmp += reference
        #         else:
        #             tmp += reference[:, :, None, :]
        #     else:
        #         assert reference.shape[-1] == 4
        #         if self.epqm:
        #             tmp += reference[..., :2]
        #         else:
        #             tmp += reference[:, :, None, :2]
        #     outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
        #     outputs_classes.append(outputs_class)
        #     outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        # outputs_anchor = torch.stack(outputs_anchors)
        outputs_anchor = anchor_point
        # outputs_coord = torch.stack(outputs_coords)
        outputs_coord = inter_references
        # outputs_seg_mask = torch.stack(outputs_seg_mask)
        outputs_seg_mask = segmentation_maps

        out = {'pred_logits': outputs_class[-1], 'pred_ctrl_points': outputs_coord[-1], \
               'pred_anchor_points':outputs_anchor[-1], 'pred_seg_mask':outputs_seg_mask[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_anchor, outputs_seg_mask)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_anchor, outputs_seg_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {'pred_logits': a, 'pred_ctrl_points': b, 'pred_anchor_points': c, 'pred_seg_mask': d} \
            for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_anchor[:-1], outputs_seg_mask[:-1])
        ]