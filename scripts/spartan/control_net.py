# https://github.com/Mikubill/sd-webui-controlnet/wiki/API#examples-1

import copy
from PIL import Image
from modules.api.api import encode_pil_to_base64
from scripts.spartan.shared import logger
import numpy as np
import json
import enum


def np_to_b64(image: np.ndarray):
    pil = Image.fromarray(image)
    image_b64 = str(encode_pil_to_base64(pil), 'utf-8')
    image_b64 = 'data:image/png;base64,' + image_b64

    return image_b64


def pack_control_net(cn_units) -> dict:
    """
    Given the control-net units, return the enveloping controlnet dict to be used with the api
    """
    controlnet = {
        'controlnet':
            {
                'args': []
            }
    }
    cn_args = controlnet['controlnet']['args']

    for i in range(0, len(cn_units)):
        if cn_units[i].enabled:
            cn_args.append(copy.deepcopy(cn_units[i].__dict__))
        else:
            logger.debug(f"controlnet unit {i} is not enabled (ignoring)")

    for i in range(0, len(cn_args)):
        unit = cn_args[i]

        # serialize image
        image_pair = unit.get('image')
        if image_pair is not None:
            image_b64 = np_to_b64(image_pair['image'])
            unit['input_image'] = image_b64  # mikubill
            unit['image'] = image_b64  # forge

            if np.all(image_pair['mask'] == 0):
                # stand-alone mask from second gradio component
                standalone_mask = unit.get('mask_image')
                if standalone_mask is not None:
                    logger.debug(f"found stand-alone mask for controlnet unit {i}")
                    mask_b64 = np_to_b64(unit['mask_image']['mask'])
                    unit['mask'] = mask_b64  # mikubill
                    unit['mask_image'] = mask_b64  # forge

            else:
                # mask from singular gradio image component
                logger.debug(f"found mask for controlnet unit {i}")
                mask_b64 = np_to_b64(image_pair['mask'])
                unit['mask'] = mask_b64  # mikubill
                unit['mask_image'] = mask_b64  # forge


        # serialize all enums
        for k in unit.keys():
            if isinstance(unit[k], enum.Enum):
                unit[k] = unit[k].value

        # avoid returning duplicate detection maps since master should return the same one
        unit['save_detected_map'] = False

    try:
        json.dumps(controlnet)
    except Exception as e:
        logger.error(f"failed to serialize controlnet\nfirst unit:\n{controlnet['controlnet']['args'][0]}")
        return {}

    return controlnet
