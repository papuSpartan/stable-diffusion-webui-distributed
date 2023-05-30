"""
https://github.com/papuSpartan/stable-diffusion-webui-distributed
"""

import base64
import io
import json
import re
import gradio
from modules import scripts
from modules import processing
from threading import Thread
from PIL import Image
from typing import List
import urllib3
import copy
from modules.images import save_image
from modules.shared import cmd_opts
import time
from pathlib import Path
import os
import subprocess
from scripts.spartan.World import World, NotBenchmarked, WorldAlreadyInitialized
from scripts.spartan.Worker import Worker, State
from modules.shared import opts
from scripts.spartan.shared import logger
from scripts.spartan.control_net import pack_control_net


# TODO implement SSDP advertisement of some sort in sdwui api to allow extension to automatically discover workers?
# TODO see if the current api has some sort of UUID generation functionality.

# noinspection PyMissingOrEmptyDocstring
class Script(scripts.Script):
    response_cache: json = None
    worker_threads: List[Thread] = []
    # Whether to verify worker certificates. Can be useful if your remotes are self-signed.
    verify_remotes = False if cmd_opts.distributed_skip_verify_remotes else True

    is_img2img = True
    is_txt2img = True
    alwayson = False
    first_run = True
    master_start = None

    world = None

    # p's type is
    # "modules.processing.StableDiffusionProcessingTxt2Img"
    # runs every time the generate button is hit

    def title(self):
        return "Distribute"

    def show(self, is_img2img):
        # return scripts.AlwaysVisible
        return True

    def ui(self, is_img2img):

        with gradio.Box():  # adds padding so our components don't look out of place
            with gradio.Accordion(label='Distributed', open=False) as main_accordian:

                with gradio.Tab('Status') as status_tab:
                    status = gradio.Textbox(elem_id='status', show_label=False)
                    status.placeholder = 'Refresh!'
                    jobs = gradio.Textbox(elem_id='jobs', label='Jobs', show_label=True)
                    jobs.placeholder = 'Refresh!'

                    refresh_status_btn = gradio.Button(value='Refresh')
                    refresh_status_btn.style(size='sm')
                    refresh_status_btn.click(Script.ui_connect_status, inputs=[], outputs=[jobs, status])

                    status_tab.select(fn=Script.ui_connect_status, inputs=[], outputs=[jobs, status])

                with gradio.Tab('Utils'):
                    refresh_checkpoints_btn = gradio.Button(value='Refresh checkpoints')
                    refresh_checkpoints_btn.style(full_width=False)
                    refresh_checkpoints_btn.click(Script.ui_connect_refresh_ckpts_btn, inputs=[], outputs=[])

                    sync_models_btn = gradio.Button(value='Synchronize models')
                    sync_models_btn.style(full_width=False)
                    sync_models_btn.click(Script.user_sync_script, inputs=[], outputs=[])

                    interrupt_all_btn = gradio.Button(value='Interrupt all', variant='stop')
                    interrupt_all_btn.style(full_width=False)
                    interrupt_all_btn.click(Script.ui_connect_interrupt_btn, inputs=[], outputs=[])

                    # redo benchmarks button
                    redo_benchmarks_btn = gradio.Button(value='Redo benchmarks', variant='stop')
                    redo_benchmarks_btn.style(full_width=False)
                    redo_benchmarks_btn.click(Script.ui_connect_benchmark_button, inputs=[], outputs=[])


        return

    @staticmethod
    def ui_connect_benchmark_button():
        logger.info("Redoing benchmarks...")
        Script.world.benchmark(rebenchmark=True)

    @staticmethod
    def user_sync_script():
        user_scripts = Path(os.path.abspath(__file__)).parent.joinpath('user')
        # user_script = user_scripts.joinpath('example.sh')
        for file in user_scripts.iterdir():
            if file.is_file() and file.name.startswith('sync'):
                user_script = file

        suffix = user_script.suffix[1:]

        if suffix == 'ps1':
            subprocess.call(['powershell', user_script])
            return True
        else:
            f = open(user_script, 'r')
            first_line = f.readline().strip()
            if first_line.startswith('#!'):
                shebang = first_line[2:]
                subprocess.call([shebang, user_script])
                return True

        return False

    # World is not constructed until the first generation job, so I use an intermediary call
    @staticmethod
    def ui_connect_interrupt_btn():
        try:
            Script.world.interrupt_remotes()
        except AttributeError:
            logger.debug("Nothing to interrupt, Distributed system not initialized")

    @staticmethod
    def ui_connect_refresh_ckpts_btn():
        try:
            Script.world.refresh_checkpoints()
        except AttributeError:
            logger.debug("Distributed system not initialized")

    @staticmethod
    def ui_connect_status():
        try:
            worker_status = ''

            for worker in Script.world.workers:
                if worker.master:
                    continue
                worker.check_connectivity() # check if worker is online
                worker_status += f"{worker.uuid} at {worker.address} is {worker.state.name}\n"

            # TODO replace this with a single check to a state flag that we should make in the world class
            for worker in Script.world.workers:
                if worker.state == State.WORKING:
                    return Script.world.__str__(), worker_status

            return 'No active jobs!', worker_status

        # init system if it isn't already
        except AttributeError as e:
            # batch size will be clobbered later once an actual request is made anyway
            Script.initialize(initial_payload=None)
            return Script.ui_connect_status()


    @staticmethod
    def add_to_gallery(processed, p):
        """adds generated images to the image gallery after waiting for all workers to finish"""

        # get master ipm by estimating based on worker speed
        master_elapsed = time.time() - Script.master_start
        logger.debug(f"Took master {master_elapsed:.2f}s")

        # wait for response from all workers
        for thread in Script.worker_threads:
            thread.join()

        for worker in Script.world.workers:
            # if it fails here then that means that the response_cache global var is not being filled for some reason
            expected_images = 1
            for job in Script.world.jobs:
                if job.worker == worker:
                    expected_images = job.batch_size

            try:
                images: json = worker.response["images"]
                # if we for some reason get more than we asked for
                if expected_images < len(images):
                    logger.debug(f"Requested {expected_images} images from '{worker.uuid}', got {len(images)}")
            except Exception:
                if worker.master is False:
                    logger.warn(f"Worker '{worker.uuid}' had nothing")
                continue

            image_params: json = worker.response["parameters"]
            image_info_post: json = json.loads(worker.response["info"])  # image info known after processing

            # visibly add work from workers to the txt2img gallery
            for i in range(0, len(images)):
                image_bytes = base64.b64decode(images[i])
                image = Image.open(io.BytesIO(image_bytes))
                processed.images.append(image)

                processed.all_prompts.append(image_params["prompt"])
                try:
                    # post-generation
                    processed.all_seeds.append(image_info_post["all_seeds"][i])
                    processed.all_subseeds.append(image_info_post["all_subseeds"][i])
                    processed.all_negative_prompts.append(image_info_post["all_negative_prompts"][i])
                except Exception as e:
                    logger.debug(f"Image at index {i} for '{worker.uuid}' was missing some post-generation data")
                    processed.all_seeds.append(image_info_post["all_seeds"][0])
                    processed.all_subseeds.append(image_info_post["all_subseeds"][0])
                    processed.all_negative_prompts.append(image_info_post["all_negative_prompts"][0])

                # generate info-text string (mostly for user use)
                this_info_text = processing.create_infotext(
                    p,
                    processed.all_prompts,
                    processed.all_seeds,
                    processed.all_subseeds,
                    comments=[""],
                    position_in_batch=i + p.batch_size,
                    iteration=0  # not sure exactly what to do with this yet
                )
                processed.infotexts.append(this_info_text)

                # save image to local disk if desired
                if cmd_opts.distributed_remotes_autosave:
                    save_image(
                        image,
                        p.outpath_samples,
                        "",
                        processed.all_seeds[i],
                        processed.all_prompts[i],
                        opts.samples_format,
                        info=this_info_text
                    )

        p.batch_size = Script.world.get_current_output_size()
        """
        This ensures that we don't get redundant outputs in a certain case: 
        We have 3 workers and we get 3 responses back.
        The user requests another 3, but due to job optimization one of the workers does not produce anything new.
        If we don't empty the response, the user will get back the two images they requested, but also one from before.
        """
        worker.response = None

        return

    @staticmethod
    def initialize(initial_payload):
        # get default batch size
        try:
            batch_size = initial_payload.batch_size
        except AttributeError:
            batch_size = 1

        if Script.world is None:
            if Script.verify_remotes is False:
                logger.warn(f"You have chosen to forego the verification of worker TLS certificates")
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            # construct World
            Script.world = World(initial_payload=initial_payload, verify_remotes=Script.verify_remotes)

            # add workers to the world
            for worker in cmd_opts.distributed_remotes:
                Script.world.add_worker(uuid=worker[0], address=worker[1], port=worker[2])

        try:
            Script.world.initialize(batch_size)
            logger.debug(f"World initialized!")
        except WorldAlreadyInitialized:
            Script.world.update_world(total_batch_size=batch_size)

    def run(self, p, *args):
        if cmd_opts.distributed_remotes is None:
            raise RuntimeError("Distributed - No remotes passed. (Try using `--distributed-remotes`?)")

        Script.initialize(initial_payload=p)

        # strip scripts that aren't yet supported and warn user
        controlnet = None
        arg = 0
        while arg < len(p.script_args) - 1:
            try:
                json.dumps(p.script_args[arg])
            except Exception:

                # find which script owns the offending arguments and fix if supported
                script_args_upper = None  # the upper index of the offending scripts args so that we can skip the rest
                for script in p.scripts.scripts:
                    title = script.title()

                    # check for supported scripts
                    if title == "ControlNet" and script.alwayson is True:
                        # grab all controlnet units
                        cn_units = []
                        for cn_arg in range(script.args_from, script.args_to + 1):
                            if isinstance(p.script_args[cn_arg], type(p.script_args[arg])):
                                cn_units.append(p.script_args[cn_arg])
                        logger.debug(f"Detected {len(cn_units)} controlnet unit(s)")

                        # get api formatted controlnet
                        controlnet: dict = pack_control_net(cn_units)

                        # ensure we don't do this more than once
                        script_args_upper = script.args_to
                        continue

                    # clean unsupported scripts
                    if script.args_from <= arg <= script.args_to:
                        logger.warn(f"Distributed does not yet support '{title}'")
                        sanitized_script_args = p.script_args[:arg] + p.script_args[arg + 1:]
                        p.script_args = sanitized_script_args

                if script_args_upper is not None:
                    arg = script_args_upper

            arg += 1

        # encapsulating the request object within a txt2imgreq object is deprecated and no longer works
        # see test/basic_features/txt2img_test.py for an example
        payload = copy.copy(p.__dict__)
        payload['batch_size'] = Script.world.get_default_worker_batch_size()
        payload['scripts'] = None
        del payload['script_args']

        if controlnet is not None:
            payload['alwayson_scripts'] = controlnet

        # TODO api for some reason returns 200 even if something failed to be set.
        #  for now we may have to make redundant GET requests to check if actually successful...
        #  https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8146
        name = re.sub(r'\s?\[[^\]]*\]$', '', opts.data["sd_model_checkpoint"])
        vae = opts.data["sd_vae"]
        option_payload = {
            # "sd_model_checkpoint": opts.data["sd_model_checkpoint"],
            "sd_model_checkpoint": name,
            "sd_vae": vae
        }

        # start generating images assigned to remote machines
        sync = False  # should only really to sync once per job
        Script.world.optimize_jobs(payload)  # optimize work assignment before dispatching
        for job in Script.world.jobs:
            if job.batch_size < 1 or job.worker.master:
                continue

            payload['batch_size'] = job.batch_size

            if job.worker.loaded_model != name or job.worker.loaded_vae != vae:
                sync = True
                job.worker.loaded_model = name
                job.worker.loaded_vae = vae

            t = Thread(target=job.worker.request, args=(payload, option_payload, sync, ))

            t.start()
            Script.worker_threads.append(t)

        # if master batch size was changed again due to optimization change it to the updated value
        p.batch_size = Script.world.get_master_batch_size()
        Script.master_start = time.time()

        available_workers = sum(1 for worker in Script.world.workers if worker.state != State.UNAVAILABLE)
        # generate images assigned to local machine
        if available_workers > 1:
            p.do_not_save_grid = True
            processed = processing.process_images(p, *args)
            Script.add_to_gallery(processed, p)
        else:
            processed = processing.process_images(p, *args)

        return processed
