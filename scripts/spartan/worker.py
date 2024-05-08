import asyncio
import base64
import copy
import io
import json
import math
import queue
import re
import time
from enum import Enum
from threading import Thread
from typing import List, Union
import requests
from modules.api.api import encode_pil_to_base64
from modules.shared import cmd_opts
from modules.shared import state as master_state
from . import shared as sh
from .shared import logger, warmup_samples, LOG_LEVEL

try:
    from webui import server_name
except ImportError:  # webui 95821f0132f5437ef30b0dbcac7c51e55818c18f and newer
    from modules.initialize_util import gradio_server_name

    server_name = gradio_server_name()
from .pmodels import Worker_Model


class InvalidWorkerResponse(Exception):
    """
    Should be raised when an invalid or unexpected response is received from a worker request.
    """
    pass


class State(Enum):
    IDLE = 1
    WORKING = 2
    INTERRUPTED = 3
    UNAVAILABLE = 4
    DISABLED = 5


class Worker:
    """
        This class represents a worker node in a distributed computing setup.

        Attributes:
            address (str): The address of the worker node. Can be an ip or a FQDN. Defaults to None.
            port (int): The port number used by the worker node. Defaults to None.
            avg_ipm (int): The average images per minute of the node. Defaults to None.
            label (str): The name of the worker node. Defaults to None.
            queried (bool): Whether this worker's memory status has been polled yet. Defaults to False.
            verify_remotes (bool): Whether to verify the validity of remote worker certificates. Defaults to False.
            master (bool): Whether this worker is the master node. Defaults to False.
            auth (str|None): The username and password used to authenticate with the worker.
            Defaults to None. (username:password)
            benchmarked (bool): Whether this worker has been benchmarked. Defaults to False.
            eta_percent_error (List[float]): A runtime list of ETA percent errors for this worker. Empty by default
            response (requests.Response): The last response from this worker. Defaults to None.

        Raises:
            InvalidWorkerResponse: If the worker responds with an invalid or unexpected response.
        """

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

    def __init__(self, address: Union[str, None] = None, port: int = 7860, label: Union[str, None] = None,
                 verify_remotes: bool = True, master: bool = False, tls: bool = False, state: State = State.IDLE,
                 avg_ipm: float = 0.0, eta_percent_error=None, user: str = None, password: str = None, pixel_cap: int = -1
                 ):

        if eta_percent_error is None:
            self.eta_percent_error = []
        else:
            self.eta_percent_error = eta_percent_error
        self.avg_ipm = avg_ipm
        self.state = state if type(state) is State else State(state)
        self.address = address
        self.port = port
        self.response_time = None
        self.loaded_model = ''
        self.loaded_vae = ''
        self.supported_scripts = {}
        self.label = label
        self.tls = tls
        self.model_override: Union[str, None] = None
        self.free_vram: int = 0
        self.response = None
        self.queried = False
        self.benchmarked = False
        self.pixel_cap = pixel_cap  # ex. limit, 2 512x512 images at once: (2*(512*512)) = 524288 px
        self.jobs_requested = 0

        # master specific setup
        if master is True:
            self.master = master
            self.label = 'master'

            # right now this is really only for clarity while debugging:
            self.address = server_name if server_name is not None else 'localhost'
            if cmd_opts.port is None:
                self.port = 7860
            else:
                self.port = cmd_opts.port
            return
        else:
            self.master = False

        # strip http:// or https:// from address if present
        if address is not None:
            if address.startswith("http://"):
                address = address[7:]
            elif address.startswith("https://"):
                address = address[8:]
                self.tls = True
                self.port = 443
            if address.endswith('/'):
                address = address[:-1]
        else:
            raise InvalidWorkerResponse("Worker address cannot be None")

        # auth
        self.user = str(user)  # casting these "prevents future issues with requests"
        self.password = str(password)

        # requests session
        self.session = requests.Session()
        self.session.auth = (self.user, self.password)
        # sometimes breaks: https://github.com/psf/requests/issues/2255
        self.session.verify = verify_remotes

    def __str__(self):
        return f"{self.address}:{self.port}"

    def __repr__(self):
        return f"'{self.label}'@{self.address}:{self.port}, speed: {self.avg_ipm} ipm, state: {self.state}"

    def __eq__(self, other):
        if isinstance(other, Worker) and other.label == self.label:
            return True
        return False

    @property
    def model(self) -> Worker_Model:
        return Worker_Model(**self.__dict__)

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
        protocol = 'http' if not self.tls else 'https'
        return f"{protocol}://{self.__str__()}/sdapi/v1/{route}"

    def eta_hr(self, payload: dict) -> float:
        """
        takes a normal payload and returns the eta of a pseudo payload which mirrors the hr-fix parameters
        This returns the eta of how long it would take to run hr-fix on the original image
        """

        pseudo_payload = copy.copy(payload)
        pseudo_payload['enable_hr'] = False  # prevent overflow in self.eta
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

        return self.eta(payload=pseudo_payload, quiet=True)

    def eta(self, payload: dict, quiet: bool = False, batch_size: int = None, samples: int = None) -> float:
        """
            estimate how long it will take to generate image(s) on a worker in seconds

            Args:
                payload: Sdwui api formatted payload
                quiet: Whether to print error correction information
                batch_size: Overrides the batch_size parameter of the payload
        """

        steps = payload['steps'] if samples is None else samples
        num_images = payload['batch_size'] if batch_size is None else batch_size

        # if worker has not yet been benchmarked then
        eta = (num_images / self.avg_ipm) * 60
        # show effect of increased step size
        real_steps_to_benched = steps / sh.benchmark_payload.steps
        eta = eta * real_steps_to_benched

        # show effect of high-res fix
        hr = payload.get('enable_hr', False)
        if hr:
            eta += self.eta_hr(payload=payload)

        # show effect of image size
        real_pix_to_benched = (payload['width'] * payload['height']) \
                              / (sh.benchmark_payload.width * sh.benchmark_payload.height)
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
                logger.warning(f"Efficiency of sampler '{payload['sampler_name']}' has not been recorded.\n")
                # in this case the sampler will be treated as having the same efficiency as Euler a

        # adjust for a known inaccuracy in our estimation of this worker using average percent error
        if len(self.eta_percent_error) > 0:
            correction = eta * (self.eta_mpe() / 100)

            if not quiet:
                logger.debug(f"worker '{self.label}'s last ETA was off by {correction:.2f}%")
            correction_summary = f"correcting '{self.label}'s ETA: {eta:.2f}s -> "
            # do regression
            eta -= correction

            if not quiet:
                correction_summary += f"{eta:.2f}s"
                logger.debug(correction_summary)

        return eta

    def request(self, payload: dict, option_payload: dict, sync_options: bool):
        """
        Sends an arbitrary amount of requests to a sdwui api depending on the context.

        Args:
            payload (dict): The txt2img payload.
            option_payload (dict): The options payload.
            sync_options (bool): Whether to attempt to synchronize the worker's loaded models with the locals'
        """
        eta = None

        try:

            if self.jobs_requested != 0:  # prevent potential hang at startup
                # if state is already WORKING then weights may be loading on worker
                # prevents issue where model override loads a large model and consecutive requests timeout
                max_wait = 30
                waited = 0
                while self.state == State.WORKING:
                    if waited >= max_wait:
                        break

                    time.sleep(1)
                    waited += 1
                if waited != 0:
                    logger.debug(f"waited {waited}s for worker '{self.label}' to IDLE before consecutive request")
                    if waited >= (0.85 * max_wait):
                        logger.warning("this seems long, so if you see this message often, consider reporting an issue")

            self.state = State.WORKING

            # query memory available on worker and store for future reference
            if self.queried is False:
                self.queried = True
                memory_response = self.session.get(
                    self.full_url("memory")
                )
                memory_response = memory_response.json()
                try:
                    memory_response = memory_response['cuda']['system']  # all in bytes
                    free_vram = int(memory_response['free']) / (1024 * 1024 * 1024)
                    total_vram = int(memory_response['total']) / (1024 * 1024 * 1024)
                    logger.debug(f"Worker '{self.label}' {free_vram:.2f}/{total_vram:.2f} GB VRAM free\n")
                    self.free_vram = memory_response['free']
                except KeyError:
                    try:
                        error = memory_response['cuda']['error']
                        msg = f"CUDA seems unavailable for worker '{self.label}'\nError: {error}"
                        logger.warning(msg)
                        # gradio.Warning("Distributed: "+msg)
                    except KeyError:
                        logger.error(f"An error occurred querying memory statistics from worker '{self.label}'\n"
                                     f"{memory_response}")

            if sync_options is True:
                self.load_options(model=option_payload['sd_model_checkpoint'], vae=option_payload['sd_vae'])

            if self.benchmarked:
                eta = self.eta(payload=payload) * payload['n_iter']
                logger.debug(f"worker '{self.label}' predicts it will take {eta:.3f}s to generate "
                             f"{payload['batch_size'] * payload['n_iter']} image(s) "
                             f"at a speed of {self.avg_ipm:.2f} ipm\n")

            try:
                # remove anything that is not serializable
                # s_tmax can be float('inf') which is not serializable, so we convert it to the max float value
                s_tmax = payload.get('s_tmax', 0.0)
                if s_tmax > 1e308:
                    payload['s_tmax'] = 1e308
                # remove unserializable caches
                payload.pop('cached_uc', None)
                payload.pop('cached_c', None)
                payload.pop('uc', None)
                payload.pop('c', None)
                payload.pop('cached_hr_c', None)
                payload.pop('cached_hr_uc', None)

                # if img2img then we need to b64 encode the init images
                init_images = payload.get('init_images', None)
                mode = 'txt2img'
                if init_images is not None:
                    mode = 'img2img'  # for use in checking script compat
                    images = []
                    for image in init_images:
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        image = 'data:image/png;base64,' + str(base64.b64encode(buffer.getvalue()), 'utf-8')
                        images.append(image)
                    payload['init_images'] = images

                alwayson_scripts = payload.get('alwayson_scripts', None)  # key may not always exist, benchmarking being one example
                if alwayson_scripts is not None:
                    if len(self.supported_scripts) <= 0:
                        payload['alwayson_scripts'] = {}
                    else:
                        matching_scripts = {}
                        missing_scripts = []
                        remote_scripts = self.supported_scripts[mode]
                        for local_script in alwayson_scripts:
                            match = False
                            for remote_script in remote_scripts:
                                if str.lower(local_script) == str.lower(remote_script):
                                    matching_scripts[local_script] = alwayson_scripts[local_script]
                                    match = True
                            if not match and str.lower(local_script) != 'distribute':
                                missing_scripts.append(local_script)

                        if len(missing_scripts) > 0:  # warn about node to node script/extension mismatching
                            message = "local script(s): "
                            for script in range(0, len(missing_scripts)):
                                message += f"\[{missing_scripts[script]}]"
                                if script < len(missing_scripts) - 1:
                                    message += ', '
                            message += f" seem to be unsupported by worker '{self.label}'\n"
                            if LOG_LEVEL == 'DEBUG':  # only warn once per session unless at debug log level
                                logger.debug(message)
                            elif self.jobs_requested < 1:
                                logger.warning(message)

                        payload['alwayson_scripts'] = matching_scripts

                # if an image mask is present
                image_mask = payload.get('image_mask', None)
                if image_mask is not None:
                    image_b64 = encode_pil_to_base64(image_mask)
                    image_b64 = str(image_b64, 'utf-8')
                    payload['mask'] = image_b64
                    del payload['image_mask']

                # see if there is anything else wrong with serializing to payload
                try:
                    json.dumps(payload)
                except Exception as e:
                    logger.error(f"Failed to serialize payload: \n{payload}")
                    # gradio.Info("Distributed: failed to serialize payload")
                    raise e

                # the main api requests sent to either the txt2img or img2img route
                response_queue = queue.Queue()

                def preemptible_request(response_queue):
                    # TODO shouldn't be this way
                    sampler_index = payload.get('sampler_index', None)
                    sampler_name = payload.get('sampler_name', None)
                    if sampler_index is None:
                        if sampler_name is not None:
                            logger.debug("had to substitute sampler index with name")
                            payload['sampler_index'] = sampler_name

                    try:
                        response = self.session.post(
                            self.full_url("txt2img") if init_images is None else self.full_url("img2img"),
                            json=payload
                        )
                        response_queue.put(response)
                    except Exception as e:
                        response_queue.put(e)  # forwarding thrown exceptions to parent thread

                request_thread = Thread(target=preemptible_request, args=(response_queue,))
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
                response = result

                self.response = response.json()
                if response.status_code != 200:
                    # try again when remote doesn't support the selected sampler by falling back to Euler a
                    if response.status_code == 404 and self.response['detail'] == "Sampler not found":
                        logger.warning(f"falling back to Euler A sampler for worker {self.label}\n"
                                       f"this may mean you should update this worker")
                        payload['sampler_index'] = 'Euler a'
                        payload['sampler_name'] = 'Euler a'

                        second_attempt = Thread(target=self.request, args=(payload, option_payload, sync_options,))
                        second_attempt.start()
                        second_attempt.join()
                        return

                    logger.error(
                        f"'{self.label}' response: Code <{response.status_code}> "
                        f"{str(response.content, 'utf-8')}")
                    self.response = None
                    raise InvalidWorkerResponse()

                # update list of ETA accuracy if state is valid
                if self.benchmarked and not self.state == State.INTERRUPTED:
                    self.response_time = time.time() - start
                    variance = ((eta - self.response_time) / self.response_time) * 100

                    logger.debug(f"Worker '{self.label}'s ETA was off by {variance:.2f}%\n"
                                 f"Predicted {eta:.2f}s. Actual: {self.response_time:.2f}s\n")

                    # if the variance is greater than 500% then we ignore it to prevent variation inflation
                    if abs(variance) < 500:
                        # check if there are already 5 samples and if so, remove the oldest
                        # this should help adjust to the user changing tasks
                        if len(self.eta_percent_error) > 4:
                            self.eta_percent_error.pop(0)
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

        except requests.RequestException:
            self.mark_unreachable()
            return

        self.state = State.IDLE
        self.jobs_requested += 1
        return

    def benchmark(self, sample_function: callable = None) -> float:
        """
        given a worker, run a small benchmark and return its performance in images/minute
        makes standard request(s) of 512x512 images and averages them to get the result
        """

        t: Thread

        if self.state in (State.DISABLED, State.UNAVAILABLE):
            logger.debug(f"worker '{self.label}' is unavailable or disabled, refusing to benchmark")
            return 0

        if self.master and sample_function is None:
            logger.critical(f"no function provided for benchmarking master")
            return -1

        def ipm(seconds: float) -> float:
            """
            Determines the rate of images per minute.

            Args:
                seconds (float): How many seconds it took to generate benchmark_payload['batch_size'] amount of images.

            Returns:
                float: Images per minute
            """

            return sh.benchmark_payload.batch_size / (seconds / 60)

        results: List[float] = []
        # it used to be lower for the first couple of generations
        # this was due to something torch does at startup according to auto and is now done at sdwui startup
        for i in range(0, sh.samples + warmup_samples):  # run some extra times so that the remote can "warm up"
            if self.state == State.UNAVAILABLE:
                self.response = None
                return 0

            try:  # if the worker is unreachable/offline then handle that here
                elapsed = None

                if not callable(sample_function):
                    start = time.time()
                    t = Thread(target=self.request, args=(dict(sh.benchmark_payload), None, False,),
                               name=f"{self.label}_benchmark_request")
                    t.start()
                    t.join()
                    elapsed = time.time() - start
                else:
                    elapsed = sample_function()

                sample_ipm = ipm(elapsed)
            except InvalidWorkerResponse as e:
                raise e

            if i >= warmup_samples:
                logger.info(f"Sample {i - warmup_samples + 1}: Worker '{self.label}'({self}) "
                            f"- {sample_ipm:.2f} image(s) per minute\n")
                results.append(sample_ipm)
            elif i == warmup_samples - 1:
                logger.debug(f"{self.label} finished warming up\n")

        # average the sample results for accuracy
        avg_ipm_result = sum(results) / sh.samples

        logger.debug(f"Worker '{self.label}' average ipm: {avg_ipm_result:.2f}")
        self.avg_ipm = avg_ipm_result
        self.response = None
        self.benchmarked = True
        self.eta_percent_error = []  # likely inaccurate after rebenching
        self.state = State.IDLE
        return avg_ipm_result

    def refresh_checkpoints(self):
        # gradio.Info("refreshing checkpoints")
        try:
            model_response = self.session.post(self.full_url('refresh-checkpoints'))
            lora_response = self.session.post(self.full_url('refresh-loras'))

            if model_response.status_code != 200:
                msg = f"Failed to refresh models for worker '{self.label}'\nCode <{model_response.status_code}>"
                logger.error(msg)
                # gradio.Warning("Distributed: "+msg)

            if lora_response.status_code != 200:
                msg = f"Failed to refresh LORA's for worker '{self.label}'\nCode <{lora_response.status_code}>"
                logger.error(msg)
                # gradio.Warning("Distributed: "+msg)
        except requests.exceptions.ConnectionError:
            self.mark_unreachable()

    def interrupt(self):
        try:
            response = self.session.post(self.full_url('interrupt'))

            if response.status_code == 200:
                self.state = State.INTERRUPTED
                logger.debug(f"successfully interrupted worker {self.label}")
        except requests.exceptions.ConnectionError:
            self.mark_unreachable()

    def reachable(self) -> bool:
        """returns false if worker is unreachable"""
        try:
            response = self.session.get(
                self.full_url("memory"),
                timeout=3
            )
            return response.status_code == 200

        except requests.exceptions.ConnectionError as e:
            logger.error(e)
            return False
        except requests.ReadTimeout as e:
            logger.critical(f"worker '{self.label}' is online but not responding (crashed?)")
            logger.error(e)
            return False

    def mark_unreachable(self):
        if self.state == State.DISABLED:
            logger.debug(f"worker '{self.label}' is disabled... refusing to mark as unavailable")
        else:
            msg = f"worker '{self.label}' at {self} was unreachable and will be avoided until reconnection"
            logger.error(msg)
            # gradio.Warning("Distributed: "+msg)
            self.state = State.UNAVAILABLE
            # invalidate models cache so that if/when worker reconnects, a new POST is sent to resync loaded models
            self.loaded_model = None
            self.loaded_vae = None

    def available_models(self) -> [List[str]]:
        if self.state == State.UNAVAILABLE or self.state == State.DISABLED or self.master:
            return []

        url = self.full_url('sd-models')
        try:
            response = self.session.get(
                url=url,
                timeout=5
            )

            if response.status_code != 200:
                logger.error(f"request to {url} returned {response.status_code}")
                if response.status_code == 404:
                    logger.error(f"did you enable --api for '{self.label}'?")
                return []

            titles = [model['title'] for model in response.json()]
            return titles
        except requests.RequestException:
            self.mark_unreachable()
            return []

    def load_options(self, model, vae=None):
        if self.master:
            return

        if self.model_override is not None:
            model = self.model_override

        model_name = re.sub(r'\s?\[[^]]*]$', '', model)
        payload = {
            "sd_model_checkpoint": model_name
        }
        if vae is not None:
            payload['sd_vae'] = vae

        self.state = State.WORKING
        start = time.time()
        response = self.session.post(
            self.full_url("options"),
            json=payload
        )
        elapsed = time.time() - start
        self.state = State.IDLE

        if response.status_code != 200:
            logger.debug(f"failed to load options for worker '{self.label}'")
        else:
            logger.debug(f"worker '{self.label}' loaded weights in {elapsed:.2f}s")
            self.loaded_model = model_name
            if vae is not None:
                self.loaded_vae = vae

            self.response = response

        return self

    def restart(self) -> bool:
        err_msg = f"could not restart worker '{self.label}'"
        success_msg = f"worker '{self.label}' is restarting"
        if self.master: # shouldn't really need to restart master (unless for convenience at some point)
            return True

        response = None
        try:
            response = self.session.post(self.full_url("server-restart"), timeout=3)
        except requests.ConnectionError:  # the successful case (kinda)
            # have to assume that the worker is actually restarting because currently sdwui does not gracefully close
            # the connection
            logger.info(success_msg)
            return True
        except requests.RequestException as e:
            logger.error(f"{err_msg}:\n{e}")
            return False

        if response.status_code == 200:
            logger.info(success_msg)
            return True
        elif response.status_code == 404:
            logger.error(f"try adding --api-server-stop to '{self.label}'s launch arguments (couldn't restart)\n"
                         "*requires webui version 1.5(5be6c02) or later")
            return False

        logger.error(f"{err_msg}: {response}")
        return False
