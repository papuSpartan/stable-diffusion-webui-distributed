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
from modules.shared import cmd_opts
import time
from scripts.spartan.World import World, WorldAlreadyInitialized
from scripts.spartan.UI import UI
from modules.shared import opts
from scripts.spartan.shared import logger
from scripts.spartan.control_net import pack_control_net
from modules.processing import fix_seed


# TODO implement SSDP advertisement of some sort in sdwui api to allow extension to automatically discover workers?
# TODO see if the current api has some sort of UUID generation functionality.

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

    if verify_remotes is False:
        logger.warning(f"You have chosen to forego the verification of worker TLS certificates")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # build world
    world = World(initial_payload=None, verify_remotes=verify_remotes)
    # add workers to the world
    for worker in cmd_opts.distributed_remotes:
        world.add_worker(uuid=worker[0], address=worker[1], port=worker[2])

    def title(self):
        return "Distribute"

    def show(self, is_img2img):
        # return scripts.AlwaysVisible
        return True

    def ui(self, is_img2img):
        extension_ui = UI(script=Script, world=Script.world)
        extension_ui.create_root()

    @staticmethod
    def add_to_gallery(processed, p):
        """adds generated images to the image gallery after waiting for all workers to finish"""

        def processed_inject_image(image, info_index, iteration: int, save_path_override=None, grid=False, response=None):
            image_params: json = response["parameters"]
            image_info_post: json = json.loads(response["info"])  # image info known after processing

            try:
                # some metadata
                processed.all_seeds.append(image_info_post["all_seeds"][info_index])
                processed.all_subseeds.append(image_info_post["all_subseeds"][info_index])
                processed.all_negative_prompts.append(image_info_post["all_negative_prompts"][info_index])
            except Exception:
                # like with controlnet masks, there isn't always full post-gen info, so we use the first images'
                logger.debug(f"Image at index {i} for '{worker.uuid}' was missing some post-generation data")
                processed_inject_image(image=image, info_index=0, iteration=iteration)
                return

            processed.all_prompts.append(image_params["prompt"])
            processed.images.append(image)  # actual received image

            # generate info-text string
            images_per_batch = p.n_iter * p.batch_size
            # zero-indexed position of image in total batch (so including master results)
            true_image_pos = len(processed.images) - 1
            num_remote_images = images_per_batch * p.batch_size
            if p.n_iter > 1:  # if splitting by batch count
                num_remote_images *= p.n_iter - 1
            info_text_used_seed_index = info_index + p.n_iter * p.batch_size if not grid else 0

            if iteration != 0:
                logger.debug(f"iteration {iteration}/{p.n_iter}, image {true_image_pos + 1}/{Script.world.total_batch_size * p.n_iter}, info-index: {info_index}, used seed index {info_text_used_seed_index}")

            info_text = processing.create_infotext(
                p=p,
                all_prompts=processed.all_prompts,
                all_seeds=processed.all_seeds,
                all_subseeds=processed.all_subseeds,
                # comments=[""], # unimplemented upstream :(
                position_in_batch=true_image_pos if not grid else 0,
                iteration=0
            )
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
        spoofed_iteration = p.n_iter
        for worker in Script.world.workers:

            expected_images = 1
            for job in Script.world.jobs:
                if job.worker == worker:
                    expected_images = job.batch_size * p.n_iter

            try:
                images: json = worker.response["images"]
                # if we for some reason get more than we asked for
                if expected_images < len(images):
                    logger.debug(f"Requested {expected_images} images from '{worker.uuid}', got {len(images)}")

                if donor_worker is None:
                    donor_worker = worker
            except Exception:
                if worker.master is False:
                    logger.warning(f"Worker '{worker.uuid}' had nothing")
                continue

            injected_to_iteration = 0
            images_per_iteration = Script.world.get_current_output_size()
            # visibly add work from workers to the image gallery
            for i in range(0, len(images)):
                image_bytes = base64.b64decode(images[i])
                image = Image.open(io.BytesIO(image_bytes))

                # inject image
                processed_inject_image(image=image, info_index=i, iteration=spoofed_iteration, response=worker.response)

                if injected_to_iteration >= images_per_iteration - 1:
                    spoofed_iteration += 1
                    injected_to_iteration = 0
                else:
                    injected_to_iteration += 1

        # generate and inject grid
        if opts.return_grid:
            grid = processing.images.image_grid(processed.images, len(processed.images))
            processed_inject_image(
                image=grid,
                info_index=0,
                save_path_override=p.outpath_grids,
                iteration=spoofed_iteration,
                grid=True,
                response=donor_worker.response
            )

        # cleanup after we're doing using all the responses
        for worker in Script.world.workers:
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

        if cmd_opts.distributed_remotes is None:
            raise RuntimeError("Distributed - No remotes passed. (Try using `--distributed-remotes`?)")

        Script.initialize(initial_payload=p)

        # strip scripts that aren't yet supported and warn user
        packed_script_args: List[dict] = []  # list of api formatted per-script argument objects
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
                # https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/12#issuecomment-1480382514
                logger.warning(f"Distributed doesn't yet support '{title}'")

        # encapsulating the request object within a txt2imgreq object is deprecated and no longer works
        # see test/basic_features/txt2img_test.py for an example
        payload = copy.copy(p.__dict__)
        payload['batch_size'] = Script.world.get_default_worker_batch_size()
        payload['scripts'] = None
        del payload['script_args']

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
        name = re.sub(r'\s?\[[^\]]*\]$', '', opts.data["sd_model_checkpoint"])
        vae = opts.data["sd_vae"]
        option_payload = {
            "sd_model_checkpoint": name,
            "sd_vae": vae
        }

        # start generating images assigned to remote machines
        sync = False  # should only really to sync once per job
        Script.world.optimize_jobs(payload)  # optimize work assignment before dispatching
        started_jobs = []
        for job in Script.world.jobs:
            payload_temp = copy.deepcopy(payload)

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
            logger.debug(f"'{job.worker.uuid}' job's given starting seed is {payload_temp['seed']} with {prior_images} coming before it")

            if job.worker.loaded_model != name or job.worker.loaded_vae != vae:
                sync = True
                job.worker.loaded_model = name
                job.worker.loaded_vae = vae

            t = Thread(target=job.worker.request, args=(payload_temp, option_payload, sync, ), name=f"{job.worker.uuid}_request")

            t.start()
            Script.worker_threads.append(t)
            started_jobs.append(job)

        # if master batch size was changed again due to optimization change it to the updated value
        p.batch_size = Script.world.get_master_batch_size()
        Script.master_start = time.time()

        # generate images assigned to local machine
        p.do_not_save_grid = True  # don't generate grid from master as we are doing this later.
        processed = processing.process_images(p, *args)
        Script.add_to_gallery(processed, p)

        return processed
