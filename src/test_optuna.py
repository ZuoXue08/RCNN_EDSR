import numpy
import torch
from model import common
import utility
import data
import torch.nn as nn
from src.option import args
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EDSR(nn.Module):
    def __init__(self, op_resblocks=30, op_feats=270, op_res_scale=0.2, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = op_resblocks
        n_feats = op_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        if args.RCNN_channel == "on":
            channels = args.n_colors + 1
        else:
            channels = args.n_colors

        # self.sub_mean = common.MeanShift(args.rgb_range)
        # Subtract the mean of the RGB channels from the image.
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        # Add the mean of the RGB channels back to the image.

        # define head module
        m_head = [conv(channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=op_res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x


loader = data.Data(args)
scale_op = args.scale
model_teop = EDSR().to(device)
model_teop.load_state_dict(torch.load('/home/6c702main/EDSR-PyTorch-optuna/src/model_r256_trial8.pt'))
torch.set_grad_enabled(False)
model_teop.eval()

with torch.no_grad():
    for idx_data, d in enumerate(loader.loader_test):
        for idx_scale, scale in enumerate(scale_op):
            d.dataset.set_scale(idx_scale)
            for lr, hr, filename in d:
                lr = lr.to(device)
                hr = hr.to(device)
                sr = model_teop(lr)
                print("test sr image:{}".format(sr))
                sr = utility.quantize(sr, rgb_range=args.rgb_range)

                psnr_optuna = 0
                psnr_optuna += utility.calc_psnr(
                    sr, hr, scale, rgb_range=args.rgb_range, dataset=d
                )
                restore_r255 = 255 / args.rgb_range
                # restore_r65536 = 65536 / args.rgb_range
                sr2 = (sr * restore_r255).squeeze().floor().cpu().numpy().astype(numpy.uint8)
                hr2 = (hr * restore_r255).squeeze().floor().cpu().numpy().astype(numpy.uint8)
                # sr2 = (sr * restore_r65536).squeeze().floor().cpu().numpy().astype(numpy.uint16)
                # hr2 = (hr * restore_r65536).squeeze().floor().cpu().numpy().astype(numpy.uint16)
                cv2.imwrite("output_hr2rc.png", hr2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite("output_sr2rc.png", sr2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print("PSNR:{}".format(psnr_optuna))
