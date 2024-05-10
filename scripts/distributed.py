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
import urllib3
from PIL import Image
from modules import processing
from modules import scripts
from modules.images import save_image
from modules.processing import fix_seed, Processed
from modules.shared import opts, cmd_opts
from modules.shared import state as webui_state
from scripts.spartan.control_net import pack_control_net
from scripts.spartan.shared import logger
from scripts.spartan.ui import UI
from scripts.spartan.world import World

old_sigint_handler = signal.getsignal(signal.SIGINT)
old_sigterm_handler = signal.getsignal(signal.SIGTERM)


# noinspection PyMissingOrEmptyDocstring
class Script(scripts.Script):

    def __init__(self):
        super().__init__()
        self.worker_threads: List[Thread] = []
        # Whether to verify worker certificates. Can be useful if your remotes are self-signed.
        self.verify_remotes = not cmd_opts.distributed_skip_verify_remotes
        self.is_img2img = True
        self.is_txt2img = True
        self.alwayson = True
        self.master_start = None
        self.runs_since_init = 0
        self.name = "distributed"
        self.is_dropdown_handler_injected = False

        if self.verify_remotes is False:
            logger.warning(f"You have chosen to forego the verification of worker TLS certificates")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # build world
        self.world = World(verify_remotes=self.verify_remotes)
        self.world.load_config()
        logger.info("doing initial ping sweep to see which workers are reachable")
        self.world.ping_remotes(indiscriminate=True)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def title(self):
        return "Distribute"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.world.load_config()
        extension_ui = UI(script=Script, world=self.world)
        # root, api_exposed = extension_ui.create_ui()
        components = extension_ui.create_ui()

        # The first injection of handler for the models dropdown(sd_model_checkpoint) which is often present
        # in the quick-settings bar of a user. Helps ensure model swaps propagate to all nodes ASAP.
        self.world.inject_model_dropdown_handler()
        # return some components that should be exposed to the api
        return components

    def add_to_gallery(self, processed, p):
        """adds generated images to the image gallery after waiting for all workers to finish"""

        def processed_inject_image(image, info_index, save_path_override=None, grid=False, response=None):
            image_params: json = response['parameters']
            image_info_post: json = json.loads(response["info"])  # image info known after processing
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
                processed_inject_image(image=image, info_index=0, response=response)
                return

            processed.all_seeds.append(seed)
            processed.all_subseeds.append(subseed)
            processed.all_negative_prompts.append(negative_prompt)
            processed.all_prompts.append(pos_prompt)
            processed.images.append(image)  # actual received image

            # generate info-text string
            # modules.ui_common -> update_generation_info renders to html below gallery
            images_per_batch = p.n_iter * p.batch_size
            # zero-indexed position of image in total batch (so including master results)
            true_image_pos = len(processed.images) - 1
            num_remote_images = images_per_batch * p.batch_size
            if p.n_iter > 1:  # if splitting by batch count
                num_remote_images *= p.n_iter - 1

            logger.debug(f"image {true_image_pos + 1}/{self.world.p.batch_size * p.n_iter}, "
                         f"info-index: {info_index}")

            if self.world.thin_client_mode:
                p.all_negative_prompts = processed.all_negative_prompts

            try:
                info_text = image_info_post['infotexts'][i]
            except IndexError:
                if not grid:
                    logger.warning(f"image {true_image_pos + 1} was missing info-text")
                info_text = processed.infotexts[0]
            info_text += f", Worker Label: {job.worker.label}"
            processed.infotexts.append(info_text)

            # automatically save received image to local disk if desired
            if cmd_opts.distributed_remotes_autosave:
                save_image(
                    image=image,
                    path=p.outpath_samples if save_path_override is None else save_path_override,
                    basename="",
                    seed=seed,
                    prompt=pos_prompt,
                    info=info_text,
                    extension=opts.samples_format
                )

        # get master ipm by estimating based on worker speed
        master_elapsed = time.time() - self.master_start
        logger.debug(f"Took master {master_elapsed:.2f}s")

        # wait for response from all workers
        webui_state.textinfo = "Distributed - receiving results"
        for thread in self.worker_threads:
            logger.debug(f"waiting for worker thread '{thread.name}'")
            thread.join()
        self.worker_threads.clear()
        logger.debug("all worker request threads returned")
        webui_state.textinfo = "Distributed - injecting images"

        # some worker which we know has a good response that we can use for generating the grid
        donor_worker = None
        for job in self.world.jobs:
            if job.batch_size < 1 or job.worker.master:
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
                processed_inject_image(image=image, info_index=i, response=job.worker.response)

        if donor_worker is None:
            logger.critical("couldn't collect any responses, the extension will have no effect")
            return

        # generate and inject grid
        if opts.return_grid:
            grid = processing.images.image_grid(processed.images, len(processed.images))
            processed_inject_image(
                image=grid,
                info_index=0,
                save_path_override=p.outpath_grids,
                grid=True,
                response=donor_worker.response
            )

        # cleanup after we're doing using all the responses
        for worker in self.world.get_workers():
            worker.response = None

        p.batch_size = len(processed.images)
        return


    # p's type is
    # "modules.processing.StableDiffusionProcessing*"
    def before_process(self, p, *args):
        if not self.world.enabled:
            logger.debug("extension is disabled")
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

            t = Thread(target=job.worker.request, args=(payload_temp, option_payload, sync,),
                       name=f"{job.worker.label}_request")

            t.start()
            self.worker_threads.append(t)
            started_jobs.append(job)

        # if master batch size was changed again due to optimization change it to the updated value
        if not self.world.thin_client_mode:
            p.batch_size = self.world.master_job().batch_size
        self.master_start = time.time()

        # generate images assigned to local machine
        p.do_not_save_grid = True  # don't generate grid from master as we are doing this later.
        self.runs_since_init += 1
        return

    def postprocess(self, p, processed, *args):
        if not self.world.enabled:
            return

        if self.master_start is not None:
            self.add_to_gallery(p=p, processed=processed)

        # restore process_images_inner if it was monkey-patched
        processing.process_images_inner = self.original_process_images_inner

    def signal_handler(self, sig, frame):
        logger.debug("handling interrupt signal")
        # do cleanup
        self.world.save_config()

        if sig == signal.SIGINT:
            if callable(old_sigint_handler):
                old_sigint_handler(sig, frame)
        else:
            if callable(old_sigterm_handler):
                old_sigterm_handler(sig, frame)
            else:
                sys.exit(0)
