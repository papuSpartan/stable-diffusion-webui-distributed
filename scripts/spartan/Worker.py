import io

import gradio
import requests
from typing import List
import math
import copy
import time
from threading import Thread
from webui import server_name
from modules.shared import cmd_opts
import gradio as gr
from scripts.spartan.shared import benchmark_payload, logger, warmup_samples
from enum import Enum
import json
import base64
import queue
from modules.shared import state as master_state


class InvalidWorkerResponse(Exception):
    """
    Should be raised when an invalid or unexpected response is received from a worker request.
    """
    pass


class State(Enum):
    IDLE = 1
    WORKING = 2
    INTERRUPTED = 3


class Worker:
    """
        This class represents a worker node in a distributed computing setup.

        Attributes:
            address (str): The address of the worker node. Can be an ip or a FQDN. Defaults to None.
            port (int): The port number used by the worker node. Defaults to None.
            avg_ipm (int): The average images per minute of the node. Defaults to None.
            uuid (str): The unique identifier/name of the worker node. Defaults to None.
            queried (bool): Whether this worker's memory status has been polled yet. Defaults to False.
            free_vram (bytes): The amount of (currently) available VRAM on the worker node. Defaults to 0.
            # TODO check this
            verify_remotes (bool): Whether to verify the validity of remote worker certificates. Defaults to False.
            master (bool): Whether this worker is the master node. Defaults to False.
            benchmarked (bool): Whether this worker has been benchmarked. Defaults to False.
            # TODO should be the last MPE from the last session
            eta_percent_error (List[float]): A runtime list of ETA percent errors for this worker. Empty by default
            last_mpe (float): The last mean percent error for this worker. Defaults to None.
            response (requests.Response): The last response from this worker. Defaults to None.
        """

    address: str = None
    port: int = None
    avg_ipm: int = None
    uuid: str = None
    queried: bool = False  # whether this worker has been connected to yet
    free_vram: bytes = 0
    verify_remotes: bool = False
    master: bool = False
    benchmarked: bool = False
    eta_percent_error: List[float] = []
    last_mpe: float = None
    response: requests.Response = None
    loaded_model: str = None
    loaded_vae: str = None
    state: State = None

    # Percentages representing (roughly) how much faster a given sampler is in comparison to Euler A.
    # We compare to euler a because that is what we currently benchmark each node with.
    other_to_euler_a = {
        "DPM++ 2S a Karras": -45.87,
        "Euler": 4.92,
        "LMS": 12.66,
        "Heun": -40.24,
        "DPM2": -42.50,
        "DPM2 a": -46.60,
        "DPM++ 2S a": -37.10,
        "DPM++ 2M": 7.46,
        "DPM++ SDE": -39.45,
        "DPM fast": 15.54,
        "DPM adaptive": -61.40,
        "LMS Karras": 5,
        "DPM2 Karras": -41,
        "DPM2 a Karras": -38.81,
        "DPM++ 2M Karras": 16.20,
        "DPM++ SDE Karras": -39.71,
        "DDIM": 0,
        "PLMS": 9.31
    }

    def __init__(self, address: str = None, port: int = None, uuid: str = None, verify_remotes: bool = None,
                 master: bool = False):
        if master is True:
            self.master = master
            self.uuid = 'master'
            # set to a sentinel value to avoid issues with speed comparisons
            self.avg_ipm = 0

            # right now this is really only for clarity while debugging:
            self.address = server_name
            if cmd_opts.port is None:
                self.port = 7860
            else:
                self.port = cmd_opts.port
            return

        self.address = address
        self.port = port
        self.verify_remotes = verify_remotes
        self.response_time = None
        self.loaded_model = ''
        self.loaded_vae = ''
        self.state = State.IDLE

        if uuid is not None:
            self.uuid = uuid

    def __str__(self):
        return f"{self.address}:{self.port}"

    def info(self, benchmark_payload) -> dict:
        """
         Stores the payload used to benchmark the world and certain attributes of the worker.
         These things are used to draw certain conclusions after the first session.

         Args:
             benchmark_payload (dict): The payload used in the benchmark.

         Returns:
             dict: Worker info, including how it was benchmarked.
         """

        d = {}
        data = {
            "avg_ipm": self.avg_ipm,
            "master": self.master,
            "benchmark_payload": benchmark_payload
        }

        d[self.uuid] = data
        return d

    def eta_mpe(self):
        """
        Returns the mean percent error using all the currently stored eta percent errors.

        Returns:
            mpe (float): The mean percent error of a worker's calculation estimates.
        """
        if len(self.eta_percent_error) == 0:
            return 0

        this_sum = 0
        for percent in self.eta_percent_error:
            this_sum += percent
        mpe = this_sum / len(self.eta_percent_error)
        return mpe

    def full_url(self, route: str) -> str:
        """
        Gets the full url used for making requests of sdwui at a given route.

        Args:
            route (str): The sdwui api route to send the request to.

        Returns:
            str: The full url.
        """

        # TODO check if using http or https
        return f"http://{self.__str__()}/sdapi/v1/{route}"

    def batch_eta_hr(self, payload: dict) -> float:
        """
        takes a normal payload and returns the eta of a pseudo payload which mirrors the hr-fix parameters
        This returns the eta of how long it would take to run hr-fix on the original image
        """

        pseudo_payload = copy.copy(payload)
        pseudo_payload['enable_hr'] = False  # prevent overflow in self.batch_eta
        res_ratio = pseudo_payload['hr_scale']
        original_steps = pseudo_payload['steps']
        second_pass_steps = pseudo_payload['hr_second_pass_steps']

        # if hires steps is set to zero then pseudo steps should = orig steps
        if second_pass_steps == 0:
            pseudo_payload['steps'] = original_steps
        else:
            pseudo_payload['steps'] = second_pass_steps

        pseudo_width = math.floor(pseudo_payload['width'] * res_ratio)
        pseudo_height = math.floor(pseudo_payload['height'] * res_ratio)
        pseudo_payload['width'] = pseudo_width
        pseudo_payload['height'] = pseudo_height

        eta = self.batch_eta(payload=pseudo_payload, quiet=True)
        return eta

    # TODO separate network latency from total eta error
    def batch_eta(self, payload: dict, quiet: bool = False) -> float:
        """estimate how long it will take to generate <batch_size> images on a worker in seconds"""
        steps = payload['steps']
        num_images = payload['batch_size']

        # if worker has not yet been benchmarked then
        try:
            eta = (num_images / self.avg_ipm) * 60
            # show effect of increased step size
            real_steps_to_benched = steps / benchmark_payload['steps']
            eta = eta * real_steps_to_benched

            # show effect of high-res fix
            hr = payload.get('enable_hr', False)
            if hr:
                eta += self.batch_eta_hr(payload=payload)

            # show effect of image size
            real_pix_to_benched = (payload['width'] * payload['height'])\
                / (benchmark_payload['width'] * benchmark_payload['height'])
            eta = eta * real_pix_to_benched

            # show effect of using a sampler other than euler a
            sampler = payload.get('sampler_name', 'Euler a')
            if sampler != 'Euler a':
                try:
                    percent_difference = self.other_to_euler_a[payload['sampler_name']]
                    if percent_difference > 0:
                        eta -= (eta * abs((percent_difference / 100)))
                    else:
                        eta += (eta * abs((percent_difference / 100)))
                except KeyError:
                    logger.warning(f"Sampler '{payload['sampler_name']}' efficiency is not recorded.\n")
                    # in this case the sampler will be treated as having the same efficiency as Euler a

            # TODO save and load each workers MPE before the end of session to workers.json.
            #  That way initial estimations are more accurate from the second sdwui session onward
            # adjust for a known inaccuracy in our estimation of this worker using average percent error
            if len(self.eta_percent_error) > 0:
                correction = eta * (self.eta_mpe() / 100)

                if not quiet:
                    logger.debug(f"worker '{self.uuid}'s last ETA was off by {correction:.2f}%")
                correction_summary = f"correcting '{self.uuid}'s ETA: {eta:.2f}s -> "
                # do regression
                eta -= correction

                if not quiet:
                    correction_summary += f"{eta:.2f}s"
                    logger.debug(correction_summary)

            return eta
        except Exception as e:
            raise e

    # TODO implement hard timeout which is independent of the requests library
    def request(self, payload: dict, option_payload: dict, sync_options: bool):
        """
        Sends an arbitrary amount of requests to a sdwui api depending on the context.

        Args:
            payload (dict): The txt2img payload.
            option_payload (dict): The options payload.
            sync_options (bool): Whether to attempt to synchronize the worker's loaded models with the locals'
        """
        eta = None

        # TODO detect remote out of memory exception and restart or garbage collect instance using api?
        try:
            self.state = State.WORKING

            # query memory available on worker and store for future reference
            if self.queried is False:
                self.queried = True
                memory_response = requests.get(
                    self.full_url("memory"),
                    verify=self.verify_remotes
                )
                memory_response = memory_response.json()['cuda']['system']  # all in bytes

                free_vram = int(memory_response['free']) / (1024 * 1024 * 1024)
                total_vram = int(memory_response['total']) / (1024 * 1024 * 1024)
                logger.debug(f"Worker '{self.uuid}' {free_vram:.2f}/{total_vram:.2f} GB VRAM free\n")
                self.free_vram = bytes(memory_response['free'])

            if sync_options is True:
                options_response = requests.post(
                    self.full_url("options"),
                    json=option_payload,
                    verify=self.verify_remotes
                )
                # TODO api returns 200 even if it fails to successfully set the checkpoint so we will have to make a
                #  second GET to see if everything loaded...

            if self.benchmarked:
                eta = self.batch_eta(payload=payload) * payload['n_iter']
                logger.debug(f"worker '{self.uuid}' predicts it will take {eta:.3f}s to generate {payload['batch_size']} image("
                      f"s) at a speed of {self.avg_ipm} ipm\n")

            try:
                # remove anything that is not serializable
                # s_tmax can be float('inf') which is not serializable, so we convert it to the max float value
                s_tmax = payload.get('s_tmax', 0.0)
                if s_tmax > 1e308:
                    payload['s_tmax'] = 1e308
                # remove cached tensor from payload as it is not serializable and not needed by the api
                payload.pop('cached_uc', None)
                # these three may be fine but the api still definitely does not need them
                payload.pop('cached_c', None)
                payload.pop('uc', None)
                payload.pop('c', None)
                # if img2img then we need to b64 encode the init images
                init_images = payload.get('init_images', None)
                if init_images is not None:
                    images = []
                    for image in init_images:
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        image = 'data:image/png;base64,' + str(base64.b64encode(buffer.getvalue()), 'utf-8')
                        images.append(image)
                    payload['init_images'] = images

                # see if there is anything else wrong with serializing to payload
                try:
                    json.dumps(payload)
                except Exception as e:
                    logger.error(f"Failed to serialize payload: \n{payload}")
                    raise e

                # the main api requests sent to either the txt2img or img2img route
                response_queue = queue.Queue()
                def preemptable_request(response_queue):
                    try:
                        response = requests.post(
                            self.full_url("txt2img") if init_images is None else self.full_url("img2img"),
                            json=payload,
                            verify=self.verify_remotes
                        )
                        response_queue.put(response)
                    except Exception as e:
                        response_queue.put(e)  # forwarding thrown exceptions to parent thread
                request_thread = Thread(target=preemptable_request, args=(response_queue,))
                interrupting = False
                start = time.time()
                request_thread.start()
                while request_thread.is_alive():
                    if interrupting is False and master_state.interrupted is True:
                        self.interrupt()
                        interrupting = True
                    time.sleep(0.5)

                result = response_queue.get()
                if isinstance(result, Exception):
                    raise result
                else:
                    response = result

                self.response = response.json()
                if response.status_code != 200:
                    logger.error(f"'{self.uuid}' response: Code <{response.status_code}> {str(response.content, 'utf-8')}")
                    self.response = None
                    raise InvalidWorkerResponse()

                # update list of ETA accuracy if state is valid
                if self.benchmarked and not self.state == State.INTERRUPTED:
                    self.response_time = time.time() - start
                    variance = ((eta - self.response_time) / self.response_time) * 100

                    logger.debug(f"\nWorker '{self.uuid}'s ETA was off by {variance:.2f}%.\n")
                    logger.debug(f"Predicted {eta:.2f}s. Actual: {self.response_time:.2f}s\n")

                    # if the variance is greater than 500% then we ignore it to prevent variation inflation
                    if abs(variance) < 500:
                        # check if there are already 5 samples and if so, remove the oldest
                        # this should help adjust to the user changing tasks
                        if len(self.eta_percent_error) > 4:
                            self.eta_percent_error.pop(0)
                        if self.eta_percent_error == 0:  # init
                            self.eta_percent_error[0] = variance
                        else:  # normal case
                            self.eta_percent_error.append(variance)
                    else:
                        logger.warning(f"Variance of {variance:.2f}% exceeds threshold of 500%. Ignoring...\n")

            except Exception as e:
                self.state = State.IDLE

                if payload['batch_size'] == 0:
                    raise InvalidWorkerResponse("Tried to request a null amount of images")
                else:
                    raise InvalidWorkerResponse(e)

        except requests.exceptions.ConnectTimeout:
            logger.error(f"\nTimed out waiting for worker '{self.uuid}' at {self}")

        self.state = State.IDLE
        return

    def benchmark(self) -> int:
        """
        given a worker, run a small benchmark and return its performance in images/minute
        makes standard request(s) of 512x512 images and averages them to get the result
        """

        t: Thread
        samples = 2  # number of times to benchmark the remote / accuracy

        logger.info(f"Benchmarking worker '{self.uuid}':\n")

        def ipm(seconds: float) -> float:
            """
            Determines the rate of images per minute.

            Args:
                seconds (float): How many seconds it took to generate benchmark_payload['batch_size'] amount of images.

            Returns:
                float: Images per minute
            """

            return benchmark_payload['batch_size'] / (seconds / 60)

        results: List[float] = []
        # it's seems to be lower for the first couple of generations
        # TODO look into how and why this "warmup" happens
        self.state = State.WORKING
        for i in range(0, samples + warmup_samples):  # run some extra times so that the remote can "warm up"
            t = Thread(target=self.request, args=(benchmark_payload, None, False,), name=f"{self.uuid}_benchmark)")
            try:  # if the worker is unreachable/offline then handle that here
                t.start()
                start = time.time()
                t.join()
                elapsed = time.time() - start
                sample_ipm = ipm(elapsed)
            except InvalidWorkerResponse as e:
                # TODO
                print(e)
                raise gr.Error(e.__str__())

            if i >= warmup_samples:
                logger.info(f"Sample {i - warmup_samples + 1}: Worker '{self.uuid}'({self}) - {sample_ipm:.2f} image(s) per "
                      f"minute\n")
                results.append(sample_ipm)
            elif i == warmup_samples - 1:
                logger.info(f"{self.uuid} warming up\n")

        # average the sample results for accuracy
        ipm_sum = 0
        for ipm in results:
            ipm_sum += ipm
        avg_ipm = math.floor(ipm_sum / samples)

        logger.info(f"Worker '{self.uuid}' average ipm: {avg_ipm}")
        self.avg_ipm = avg_ipm
        # noinspection PyTypeChecker
        self.response = None
        self.benchmarked = True
        self.state = State.IDLE
        return avg_ipm

    def refresh_checkpoints(self):
        model_response = requests.post(
            self.full_url('refresh-checkpoints'),
            json={},
            verify=self.verify_remotes
        )
        lora_response = requests.post(
            self.full_url('refresh-loras'),
            json={},
            verify=self.verify_remotes
        )

        if model_response.status_code != 200:
            logger.error(f"Failed to refresh models for worker '{self.uuid}'\nCode <{model_response.status_code}>")

        if lora_response.status_code != 200:
            logger.error(f"Failed to refresh LORA's for worker '{self.uuid}'\nCode <{lora_response.status_code}>")

    def interrupt(self):
        response = requests.post(
            self.full_url('interrupt'),
            json={},
            verify=self.verify_remotes
        )

        if response.status_code == 200:
            self.state = State.INTERRUPTED
            logger.debug(f"successfully interrupted worker {self.uuid}")
