"""
https://github.com/papuSpartan/stable-diffusion-webui-distributed
"""

import base64
import io
import json
import re

from modules import scripts, script_callbacks
from modules import processing
from threading import Thread
from PIL import Image
from typing import List
from modules.processing import StableDiffusionProcessingTxt2Img
import urllib3
import copy
from modules.images import save_image
from modules.shared import cmd_opts
import time
from scripts.spartan.World import World, Worker, NotBenchmarked, WorldAlreadyInitialized
from modules.shared import opts

path_root = scripts.basedir()


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

    @staticmethod
    def add_to_gallery(processed, p):
        """adds generated images to the image gallery after waiting for all workers to finish"""
        # get master ipm by estimating based on worker speed
        global worker
        master_elapsed = time.time() - Script.master_start
        print(f"Took master {master_elapsed}s")

        # wait for response from all workers
        for thread in Script.worker_threads:
            thread.join()

        for worker in Script.world.workers:
            # if it fails here then that means that the response_cache global var is not being filled for some reason
            try:
                images: json = worker.response["images"]
            except TypeError:
                if worker.master is False:
                    print(f"Worker '{worker.uuid}' had nothing")
                continue

            image_params: json = worker.response["parameters"]
            image_info_post: json = json.loads(worker.response["info"])  # image info known after processing

            # visibly add work from workers to the txt2img gallery
            for i in range(0, len(images)):
                image_bytes = base64.b64decode(images[i])
                image = Image.open(io.BytesIO(image_bytes))
                processed.images.append(image)

                # params
                processed.all_prompts.append(image_params["prompt"])
                # for k in vars(processed):
                #     try:
                #         if image_params[k] is not None:
                #             print(f"processed: '{processed.k}'\nparams: '{image_params[k]}'\n")
                #     except Exception as e:
                #         print(e)

                # post-generation
                processed.all_seeds.append(image_info_post["all_seeds"][i])
                processed.all_subseeds.append(image_info_post["all_subseeds"][i])
                processed.all_negative_prompts.append(image_info_post["all_negative_prompts"][i])

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
                # TODO add command line toggle for having worker results saved to disk
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

        Script.unregister_callbacks()
        return

    def run(self, p, *args):
        if cmd_opts.distributed_remotes is None:
            raise RuntimeError("Distributed - No remotes passed. (Try using `--distributed-remotes`?)")

        Script.world = World(initial_payload=p, verify_remotes=Script.verify_remotes)
        # add workers to the world
        for worker in cmd_opts.distributed_remotes:
            Script.world.add_worker(uuid=worker[0], address=worker[1], port=worker[2])
        # register gallery callback
        script_callbacks.on_after_image_processed(Script.add_to_gallery)

        if self.verify_remotes is False:
            print(f"WARNING: you have chosen to forego the verification of worker TLS certificates")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            Script.world.initialize(p.batch_size)
            print("World initialized!")
        except WorldAlreadyInitialized:
            Script.world.update_world(p.batch_size)

        # encapsulating the request object within a txt2imgreq object is deprecated and no longer works
        # see test/basic_features/txt2img_test.py for an example
        payload = p.__dict__
        payload['batch_size'] = Script.world.get_default_worker_batch_size()
        payload['scripts'] = None
        # print(payload)
        # print(opts.dumpjson())

        # TODO api for some reason returns 200 even if something failed to be set.
        #  for now we may have to make redundant GET requests to check if actually successful...
        #  https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8146
        name = re.sub(r'\s?\[[^\]]*\]$', '', opts.data["sd_model_checkpoint"])
        option_payload = {
            # "sd_model_checkpoint": opts.data["sd_model_checkpoint"],
            "sd_model_checkpoint": name,
            "sd_vae": opts.data["sd_vae"]
        }

        first_option_sync = True  # should only really to sync models once per total job
        Script.world.optimize_jobs(payload)  # optimize work assignment before dispatching
        for job in Script.world.jobs:
            if job.batch_size < 1 or job.worker.master:
                continue

            new_payload = copy.copy(payload)  # prevent race condition instead of sharing the payload object
            new_payload['batch_size'] = job.batch_size

            # print(f"requesting {new_payload['batch_size']} images from worker '{job.worker.uuid}'\n")
            t = Thread(target=job.worker.request, args=(new_payload, option_payload, first_option_sync,))
            first_option_sync = False

            t.start()
            Script.worker_threads.append(t)

        # if master batch size was changed again due to optimization change it to the updated value
        p.batch_size = Script.world.get_master_batch_size()
        Script.master_start = time.time()
        # return processing.process_images(p, *args)

    @staticmethod
    def unregister_callbacks():
        script_callbacks.remove_current_script_callbacks()


# not actually called when user selects a different script in the ui dropdown
script_callbacks.on_script_unloaded(Script.unregister_callbacks)
