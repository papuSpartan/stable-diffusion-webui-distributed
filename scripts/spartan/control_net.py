import copy
from PIL import Image
from modules.api.api import encode_pil_to_base64
from scripts.spartan.shared import logger


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
        # copy control net unit to payload
        cn_args.append(copy.copy(cn_units[i].__dict__))
        unit = cn_args[i]

        # if unit isn't enabled then don't bother including
        if not unit['enabled']:
            del unit['input_mode']
            del unit['image']
            logger.debug(f"Controlnet unit {i} is not enabled. Ignoring")
            continue

        # serialize image
        if unit['image'] is not None:
            image = unit['image']['image']
            # mask = unit['image']['mask']
            pil = Image.fromarray(image)
            image_b64 = encode_pil_to_base64(pil)
            image_b64 = str(image_b64, 'utf-8')
            unit['input_image'] = image_b64

        # remove anything unserializable
        del unit['input_mode']
        del unit['image']

    return controlnet
