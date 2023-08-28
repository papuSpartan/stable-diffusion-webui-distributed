"""
https://github.com/papuSpartan/stable-diffusion-webui-distributed
"""

import base64
import io
import json
import re
from modules import scripts
from modules import processing
from threading import Thread, current_thread
from PIL import Image
from typing import List
import urllib3
import copy
from modules.images import save_image
from modules.shared import opts, cmd_opts
from modules.shared import state as webui_state
import time
from scripts.spartan.World import World, WorldAlreadyInitialized
from scripts.spartan.UI import UI
from scripts.spartan.shared import logger
from scripts.spartan.control_net import pack_control_net
from modules.processing import fix_seed, Processed
import signal
import sys

old_sigint_handler = signal.getsignal(signal.SIGINT)
old_sigterm_handler = signal.getsignal(signal.SIGTERM)


# TODO implement advertisement of some sort in sdwui api to allow extension to automatically discover workers?
# noinspection PyMissingOrEmptyDocstring
class Script(scripts.Script):
    worker_threads: List[Thread] = []
    # Whether to verify worker certificates. Can be useful if your remotes are self-signed.
    verify_remotes = False if cmd_opts.distributed_skip_verify_remotes else True

    is_img2img = True
    is_txt2img = True
    alwayson = False
    first_run = True
    master_start = None
    runs_since_init = 0
    name = "distributed"

    if verify_remotes is False:
        logger.warning(f"You have chosen to forego the verification of worker TLS certificates")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # build world
    world = World(initial_payload=None, verify_remotes=verify_remotes)
    # add workers to the world
    world.load_config()
    if cmd_opts.distributed_remotes is not None and len(cmd_opts.distributed_remotes) > 0:
        logger.warning(f"--distributed-remotes is deprecated and may be removed in the future\n"
                       "gui/external modification of {world.config_path} will be prioritized going forward")

        for worker in cmd_opts.distributed_remotes:
            world.add_worker(uuid=worker[0], address=worker[1], port=worker[2], tls=False)
        world.save_config()
    # do an early check to see which workers are online
    logger.info("doing initial ping sweep to see which workers are reachable")
    world.ping_remotes(indiscriminate=True)

    def title(self):
        return "Distribute"

    def show(self, is_img2img):
        # return scripts.AlwaysVisible
        return True

    def ui(self, is_img2img):
        extension_ui = UI(script=Script, world=Script.world)
        root, api_exposed = extension_ui.create_ui()

        # return some components that should be exposed to the api
        return api_exposed

    @staticmethod
    def add_to_gallery(processed, p):
        """adds generated images to the image gallery after waiting for all workers to finish"""
        webui_state.textinfo = "Distributed - injecting images"

        def processed_inject_image(image, info_index, save_path_override=None, grid=False, response=None):
            image_params: json = response['parameters']
            image_info_post: json = json.loads(response["info"])  # image info known after processing
            num_response_images = image_params["batch_size"] * image_params["n_iter"]

            seed = None
            subseed = None
            negative_prompt = None

            try:
                if num_response_images > 1:
                    seed = image_info_post['all_seeds'][info_index]
                    subseed = image_info_post['all_subseeds'][info_index]
                    negative_prompt = image_info_post['all_negative_prompts'][info_index]
                else:
                    seed = image_info_post['seed']
                    subseed = image_info_post['subseed']
                    negative_prompt = image_info_post['negative_prompt']
            except IndexError:
                # like with controlnet masks, there isn't always full post-gen info, so we use the first images'
                logger.debug(f"Image at index {i} for '{job.worker.label}' was missing some post-generation data")
                processed_inject_image(image=image, info_index=0, response=response)
                return

            processed.all_seeds.append(seed)
            processed.all_subseeds.append(subseed)
            processed.all_negative_prompts.append(negative_prompt)
            processed.all_prompts.append(image_params["prompt"])
            processed.images.append(image)  # actual received image

            # generate info-text string
            # modules.ui_common -> update_generation_info renders to html below gallery
            images_per_batch = p.n_iter * p.batch_size
            # zero-indexed position of image in total batch (so including master results)
            true_image_pos = len(processed.images) - 1
            num_remote_images = images_per_batch * p.batch_size
            if p.n_iter > 1:  # if splitting by batch count
                num_remote_images *= p.n_iter - 1

            logger.debug(f"image {true_image_pos + 1}/{Script.world.total_batch_size * p.n_iter}, "
                         f"info-index: {info_index}")

            if Script.world.thin_client_mode:
                p.all_negative_prompts = processed.all_negative_prompts

            try:
                info_text = image_info_post['infotexts'][i]
            except IndexError:
                if not grid:
                    logger.warning(f"image {true_image_pos + 1} was missing info-text")
                info_text = processed.infotexts[0]
            processed.infotexts.append(info_text)

            # automatically save received image to local disk if desired
            if cmd_opts.distributed_remotes_autosave:
                save_image(
                    image=image,
                    path=p.outpath_samples if save_path_override is None else save_path_override,
                    basename="",
                    seed=processed.all_seeds[-1],
                    prompt=processed.all_prompts[-1],
                    info=info_text,
                    extension=opts.samples_format
                )

        # get master ipm by estimating based on worker speed
        master_elapsed = time.time() - Script.master_start
        logger.debug(f"Took master {master_elapsed:.2f}s")

        # wait for response from all workers
        for thread in Script.worker_threads:
            logger.debug(f"waiting for worker thread '{thread.name}'")
            thread.join()
        Script.worker_threads.clear()
        logger.debug("all worker request threads returned")

        # some worker which we know has a good response that we can use for generating the grid
        donor_worker = None
        for job in Script.world.jobs:
            if job.batch_size < 1 or job.worker.master:
                continue

            try:
                images: json = job.worker.response["images"]
                # if we for some reason get more than we asked for
                if (job.batch_size * p.n_iter) < len(images):
                    logger.debug(f"Requested {job.batch_size} image(s) from '{job.worker.label}', got {len(images)}")

                if donor_worker is None:
                    donor_worker = job.worker
            except KeyError:
                if job.batch_size > 0:
                    logger.warning(f"Worker '{job.worker.label}' had no images")
                continue
            except TypeError as e:
                if job.worker.response is None:
                    logger.error(f"worker '{job.worker.label}' had no response")
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
            logger.critical("couldn't collect any responses, distributed will do nothing")
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
        for worker in Script.world.get_workers():
            worker.response = None

        p.batch_size = len(processed.images)
        return

    @staticmethod
    def initialize(initial_payload):
        # get default batch size
        try:
            batch_size = initial_payload.batch_size
        except AttributeError:
            batch_size = 1

        try:
            Script.world.initialize(batch_size)
            logger.debug(f"World initialized!")
        except WorldAlreadyInitialized:
            Script.world.update_world(total_batch_size=batch_size)

    # p's type is
    # "modules.processing.StableDiffusionProcessingTxt2Img"
    # runs every time the generate button is hit
    def run(self, p, *args):
        current_thread().name = "distributed_main"
        Script.initialize(initial_payload=p)

        # strip scripts that aren't yet supported and warn user
        packed_script_args: List[dict] = []  # list of api formatted per-script argument objects
        # { "script_name": { "args": ["value1", "value2", ...] }
        incompat_list = []
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
                    if type(cn_arg).__name__ == "UiControlNetUnit":
                        cn_units.append(cn_arg)
                logger.debug(f"Detected {len(cn_units)} controlnet unit(s)")

                # get api formatted controlnet
                packed_script_args.append(pack_control_net(cn_units))

                continue
            else:
                # other scripts to pack
                # args_script_pack = {}
                # args_script_pack[title] = {"args": []}
                # for arg in p.script_args[script.args_from:script.args_to]:
                #     args_script_pack[title]["args"].append(arg)
                # packed_script_args.append(args_script_pack)
                # https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/12#issuecomment-1480382514
                if Script.runs_since_init < 1:
                    incompat_list.append(title)

        if Script.runs_since_init < 1 and len(incompat_list) >= 1:
            m = "Distributed doesn't yet support:"
            for i in range(0, len(incompat_list)):
                m += f" {incompat_list[i]}"
                if i < len(incompat_list) - 1:
                    m += ","
            logger.warning(m)

        # encapsulating the request object within a txt2imgreq object is deprecated and no longer works
        # see test/basic_features/txt2img_test.py for an example
        payload = copy.copy(p.__dict__)
        payload['batch_size'] = Script.world.default_batch_size()
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
        vae = opts.data["sd_vae"]
        option_payload = {
            "sd_model_checkpoint": name,
            "sd_vae": vae
        }

        # start generating images assigned to remote machines
        sync = False  # should only really need to sync once per job
        Script.world.optimize_jobs(payload)  # optimize work assignment before dispatching
        started_jobs = []

        # check if anything even needs to be done
        if len(Script.world.jobs) == 1 and Script.world.jobs[0].worker.master:
            logger.debug(f"distributed doesn't have to do anything, returning control to webui")
            return

        for job in Script.world.jobs:
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
            payload_temp['subseed'] += prior_images
            payload_temp['seed'] += prior_images if payload_temp['subseed_strength'] == 0 else 0
            logger.debug(
                f"'{job.worker.label}' job's given starting seed is "
                f"{payload_temp['seed']} with {prior_images} coming before it")

            if job.worker.loaded_model != name or job.worker.loaded_vae != vae:
                sync = True
                job.worker.loaded_model = name
                job.worker.loaded_vae = vae

            t = Thread(target=job.worker.request, args=(payload_temp, option_payload, sync,),
                       name=f"{job.worker.label}_request")

            t.start()
            Script.worker_threads.append(t)
            started_jobs.append(job)

        # if master batch size was changed again due to optimization change it to the updated value
        if not self.world.thin_client_mode:
            p.batch_size = Script.world.master_job().batch_size
        Script.master_start = time.time()

        # generate images assigned to local machine
        p.do_not_save_grid = True  # don't generate grid from master as we are doing this later.
        if Script.world.thin_client_mode:
            p.batch_size = 0
            processed = Processed(p=p, images_list=[])
            processed.all_prompts = []
            processed.all_seeds = []
            processed.all_subseeds = []
            processed.all_negative_prompts = []
            processed.infotexts = []
            processed.prompt = None
        else:
            processed = processing.process_images(p)

        Script.add_to_gallery(processed, p)
        Script.runs_since_init += 1
        return processed

    @staticmethod
    def signal_handler(sig, frame):
        logger.debug("handling interrupt signal")
        # do cleanup
        Script.world.save_config()

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
