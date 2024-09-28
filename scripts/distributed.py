"""
https://github.com/papuSpartan/stable-diffusion-webui-distributed
"""

import base64
import copy
import io
import json
import re
import signal
import sys
import time
from threading import Thread
from typing import List
import gradio
from torchvision.transforms import ToTensor
import urllib3
from PIL import Image
from modules import processing
from modules import scripts
from modules.images import save_image, image_grid
from modules.processing import fix_seed
from modules.shared import opts, cmd_opts
from modules.shared import state as webui_state
from scripts.spartan.control_net import pack_control_net
from scripts.spartan.shared import logger
from scripts.spartan.ui import UI
from scripts.spartan.world import World, State

old_sigint_handler = signal.getsignal(signal.SIGINT)
old_sigterm_handler = signal.getsignal(signal.SIGTERM)


# noinspection PyMissingOrEmptyDocstring
class DistributedScript(scripts.Script):
    # global old_sigterm_handler, old_sigterm_handler
    # Whether to verify worker certificates. Can be useful if your remotes are self-signed.
    verify_remotes = not cmd_opts.distributed_skip_verify_remotes
    master_start = None
    runs_since_init = 0
    name = "distributed"
    is_dropdown_handler_injected = False

    if verify_remotes is False:
        logger.warning(f"You have chosen to forego the verification of worker TLS certificates")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # build world
    world = World(verify_remotes=verify_remotes)
    world.load_config()
    logger.info("doing initial ping sweep to see which workers are reachable")
    world.ping_remotes(indiscriminate=True)

    # constructed for both txt2img and img2img
    def __init__(self):
        super().__init__()

    def title(self):
        return "Distribute"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        extension_ui = UI(world=self.world, is_img2img=is_img2img)
        # root, api_exposed = extension_ui.create_ui()
        components = extension_ui.create_ui()

        # The first injection of handler for the models dropdown(sd_model_checkpoint) which is often present
        # in the quick-settings bar of a user. Helps ensure model swaps propagate to all nodes ASAP.
        self.world.inject_model_dropdown_handler()
        # return some components that should be exposed to the api
        return components

    def add_to_gallery(self, pp, p):
        """adds generated images to the image gallery after waiting for all workers to finish"""

        def processed_inject_image(image, info_index, save_path_override=None, grid=False, job=None):
            image_params: json = job.worker.response['parameters']
            image_info_post: json = json.loads(job.worker.response["info"])  # image info known after processing
            num_response_images = image_params["batch_size"] * image_params["n_iter"]

            seed = None
            subseed = None
            negative_prompt = None
            pos_prompt = None

            try:
                if num_response_images > 1:
                    seed = image_info_post['all_seeds'][info_index]
                    subseed = image_info_post['all_subseeds'][info_index]
                    negative_prompt = image_info_post['all_negative_prompts'][info_index]
                    pos_prompt = image_info_post['all_prompts'][info_index]
                else:
                    seed = image_info_post['seed']
                    subseed = image_info_post['subseed']
                    negative_prompt = image_info_post['negative_prompt']
                    pos_prompt = image_info_post['prompt']
            except IndexError:
                # like with controlnet masks, there isn't always full post-gen info, so we use the first images'
                logger.debug(f"Image at index {i} for '{job.worker.label}' was missing some post-generation data")
                processed_inject_image(image=image, info_index=0, job=job)
                return

            p.seeds.append(seed)
            p.subseeds.append(subseed)
            p.negative_prompts.append(negative_prompt)
            p.prompts.append(pos_prompt)

            transform = ToTensor()
            pp.images.append(transform(image))  # actual received image

            # modules.ui_common -> update_generation_info renders to html below gallery
            images_per_batch = p.n_iter * p.batch_size
            # zero-indexed position of image in gallery (so including master/local results)
            true_image_pos = (len(pp.images) - 1) + (not p.do_not_save_grid)
            num_remote_images = images_per_batch * p.batch_size
            if p.n_iter > 1:  # if splitting by batch count
                num_remote_images *= p.n_iter - 1

            logger.debug(f"image {true_image_pos + 1}/{self.world.p.batch_size * p.n_iter}, "
                         f"info-index: {info_index}")

            if self.world.thin_client_mode:
                p.all_negative_prompts = pp.all_negative_prompts

            # saves final position of image in gallery so that we can later modify the correct infotext
            job.gallery_map.append(true_image_pos)

        # get master ipm by estimating based on worker speed
        master_elapsed = time.time() - self.master_start
        logger.debug(f"Took master {master_elapsed:.2f}s")

        # wait for response from all workers
        webui_state.textinfo = "Distributed - receiving results"
        for job in self.world.jobs:
            if job.thread is None:
                continue

            logger.debug(f"waiting for worker thread '{job.thread.name}'")
            job.thread.join()
        logger.debug("all worker request threads returned")
        webui_state.textinfo = "Distributed - injecting images"

        # some worker which we know has a good response that we can use for generating the grid
        donor_worker = None
        for job in self.world.jobs:
            if job.worker.response is None or job.batch_size < 1 or job.worker.master:
                continue

            try:
                images: json = job.worker.response["images"]
                # if we for some reason get more than we asked for
                if (job.batch_size * p.n_iter) < len(images):
                    logger.debug(f"requested {job.batch_size} image(s) from '{job.worker.label}', got {len(images)}")

                if donor_worker is None:
                    donor_worker = job.worker
            except KeyError:
                if job.batch_size > 0:
                    logger.warning(f"Worker '{job.worker.label}' had no images")
                continue
            except TypeError as e:
                if job.worker.response is None:
                    msg = f"worker '{job.worker.label}' had no response"
                    logger.error(msg)
                    gradio.Warning("Distributed: "+msg)
                else:
                    logger.exception(e)
                continue

            # visibly add work from workers to the image gallery
            for i in range(0, len(images)):
                image_bytes = base64.b64decode(images[i])
                image = Image.open(io.BytesIO(image_bytes))

                # inject image
                processed_inject_image(image=image, info_index=i, job=job)

        if donor_worker is None:
            logger.critical("couldn't collect any responses, the extension will have no effect")
            return


        p.batch_size = len(pp.images)
        webui_state.textinfo = ""
        return

    # p's type is
    # "modules.processing.StableDiffusionProcessing*"
    def before_process(self, p, *args):
        is_img2img = getattr(p, 'init_images', False)
        if is_img2img and self.world.enabled_i2i is False:
            logger.debug("extension is disabled for i2i")
            return
        elif not is_img2img and self.world.enabled is False:
            logger.debug("extension is disabled for t2i")
            return
        self.world.update(p)

        # save original process_images_inner function for later if we monkeypatch it
        self.original_process_images_inner = processing.process_images_inner

        # strip scripts that aren't yet supported and warn user
        packed_script_args: List[dict] = []  # list of api formatted per-script argument objects
        # { "script_name": { "args": ["value1", "value2", ...] }
        for script in p.scripts.scripts:
            if script.alwayson is not True:
                continue
            title = script.title()

            if title == "ADetailer":
                adetailer_args = p.script_args[script.args_from:script.args_to]

                # InputAccordion main toggle, skip img2img toggle
                if adetailer_args[0] and adetailer_args[1]:
                    logger.debug(f"adetailer is skipping img2img, returning control to wui")
                    return

            # check for supported scripts
            if title == "ControlNet":
                # grab all controlnet units
                cn_units = []
                cn_args = p.script_args[script.args_from:script.args_to]

                for cn_arg in cn_args:
                    if "ControlNetUnit" in type(cn_arg).__name__:
                        cn_units.append(cn_arg)
                logger.debug(f"Detected {len(cn_units)} controlnet unit(s)")

                # get api formatted controlnet
                packed_script_args.append(pack_control_net(cn_units))

                continue

            # other scripts to pack
            args_script_pack = {title: {"args": []}}
            for arg in p.script_args[script.args_from:script.args_to]:
                args_script_pack[title]["args"].append(arg)
            packed_script_args.append(args_script_pack)
            # https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/12#issuecomment-1480382514

        # encapsulating the request object within a txt2imgreq object is deprecated and no longer works
        # see test/basic_features/txt2img_test.py for an example
        payload = copy.copy(p.__dict__)
        payload['batch_size'] = self.world.default_batch_size()
        payload['scripts'] = None
        try:
            del payload['script_args']
        except KeyError:
            del payload['script_args_value']

        payload['alwayson_scripts'] = {}
        for packed in packed_script_args:
            payload['alwayson_scripts'].update(packed)

        # generate seed early for master so that we can calculate the successive seeds for each slave
        fix_seed(p)
        payload['seed'] = p.seed
        payload['subseed'] = p.subseed

        # TODO api for some reason returns 200 even if something failed to be set.
        #  for now we may have to make redundant GET requests to check if actually successful...
        #  https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8146

        name = re.sub(r'\s?\[[^]]*]$', '', opts.data["sd_model_checkpoint"])
        vae = opts.data.get('sd_vae')
        option_payload = {
            "sd_model_checkpoint": name,
            "sd_vae": vae
        }

        # start generating images assigned to remote machines
        sync = False  # should only really need to sync once per job
        self.world.optimize_jobs(payload)  # optimize work assignment before dispatching
        started_jobs = []

        # check if anything even needs to be done
        if len(self.world.jobs) == 1 and self.world.jobs[0].worker.master:

            if payload['batch_size'] >= 2:
                msg = f"all remote workers are offline or unreachable"
                gradio.Info(f"Distributed: "+msg)
                logger.critical(msg)

            logger.debug(f"distributed has nothing to do, returning control to webui")

            return

        for job in self.world.jobs:
            if job.worker.state in (State.UNAVAILABLE, State.DISABLED):
                continue

            payload_temp = copy.copy(payload)
            del payload_temp['scripts_value']
            payload_temp = copy.deepcopy(payload_temp)

            if job.worker.master:
                started_jobs.append(job)
            if job.batch_size < 1 or job.worker.master:
                continue

            prior_images = 0
            for j in started_jobs:
                prior_images += j.batch_size * p.n_iter

            payload_temp['batch_size'] = job.batch_size
            if job.step_override is not None:
                payload_temp['steps'] = job.step_override
            payload_temp['subseed'] += prior_images
            payload_temp['seed'] += prior_images if payload_temp['subseed_strength'] == 0 else 0
            logger.debug(
                f"'{job.worker.label}' job's given starting seed is "
                f"{payload_temp['seed']} with {prior_images} coming before it"
            )

            if job.worker.loaded_model != name or job.worker.loaded_vae != vae:
                sync = True
                job.worker.loaded_model = name
                job.worker.loaded_vae = vae

            job.thread = Thread(target=job.worker.request, args=(payload_temp, option_payload, sync,),
                       name=f"{job.worker.label}_request")
            job.thread.start()
            started_jobs.append(job)

        # if master batch size was changed again due to optimization change it to the updated value
        if not self.world.thin_client_mode:
            p.batch_size = self.world.master_job().batch_size
        self.master_start = time.time()

        # generate images assigned to local machine
        # p.do_not_save_grid = True  # don't generate grid from master as we are doing this later.
        self.runs_since_init += 1
        return

    def postprocess_batch_list(self, p, pp, *args, **kwargs):
        is_img2img = getattr(p, 'init_images', False)
        if is_img2img and self.world.enabled_i2i is False:
            return
        elif not is_img2img and self.world.enabled is False:
            return

        if self.master_start is not None:
            self.add_to_gallery(p=p, pp=pp)


    def postprocess(self, p, processed, *args):
        for job in self.world.jobs:
            if job.worker.response is not None:
                for i, v in enumerate(job.gallery_map):
                    infotext = json.loads(job.worker.response['info'])['infotexts'][i]
                    logger.debug(f"replacing image {v}'s infotext with\n"
                                    f"> '{infotext}'")
                    infotext += f", Worker Label: {job.worker.label}"
                    processed.infotexts[v] = infotext

        # cleanup
        for worker in self.world.get_workers():
            worker.response = None
        # restore process_images_inner if it was monkey-patched
        processing.process_images_inner = self.original_process_images_inner
        # save any dangling state to prevent load_config in next iteration overwriting it
        self.world.save_config()

    @staticmethod
    def signal_handler(sig, frame):
        logger.debug("handling interrupt signal")
        # do cleanup
        DistributedScript.world.save_config()

        if sig == signal.SIGINT:
            if callable(old_sigint_handler):
                old_sigint_handler(sig, frame)
        else:
            if callable(old_sigterm_handler):
                old_sigterm_handler(sig, frame)
            else:
                sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
