import os
import numpy as np
from PIL import Image
import sys
import torch
import torch.nn.functional as F

class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f'{frame_id}.png')
            return np.array(Image.open(mask_path))
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write("The frames have to be indexed PNG files placed inside the corespondent sequence "
                             "folder.\nThe indexes have to match with the initial frame.\n")
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def load_attn(self, sequence, h, w):
        masks = torch.load(os.path.join('DAVIS_Attn', sequence+'.pth'), map_location='cpu')
        max_obj = 0
        results = []
        for t, v in masks.items():
            mask = v[0][0]
            max_obj = max(max_obj, mask.shape[0])
            mask = F.interpolate(mask, (h, w))
            mask = mask[:, 0]
            num_obj = mask.shape[0]
            mask = mask.view(num_obj, h, w)
            mask = (mask > 0.3)
            results.append(mask)
        results.append(results[-1])
        for i, mask in enumerate(results):
            if mask.shape[0] < max_obj:
                new = torch.zeros(max_obj, *mask.shape[1:])
                new[:mask.shape[0]] = mask
                results[i] = new
        results = torch.stack(results, dim=1)
        results = results.numpy()
        return results

    def read_masks(self, sequence, masks_id):
        mask_0 = self._read_mask(sequence, masks_id[0])
        masks = self.load_attn(sequence, *mask_0.shape)
        return masks
