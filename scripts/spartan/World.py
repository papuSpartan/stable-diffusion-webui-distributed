"""
This module facilitates the creation of a stable-diffusion-webui centered distributed computing system.

World:
    The main class which should be instantiated in order to create a new sdwui distributed system.
"""

import copy
import json
import math
import os
import time
from typing import List
from threading import Thread
import requests
from inspect import getsourcefile
from os.path import abspath
from pathlib import Path
from modules.processing import process_images

# from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img

benchmark_payload: dict = {
    "prompt": "A herd of cows grazing at the bottom of a sunny valley",
    "negative_prompt": "",
    "steps": 20,
    "width": 512,
    "height": 512,
    "batch_size": 1
}


class NotBenchmarked(Exception):
    """
    Should be raised when attempting to do something that requires knowledge of worker benchmark statistics, and
    they haven't been calculated yet.
    """
    pass


class InvalidWorkerResponse(Exception):
    """
    Should be raised when an invalid or unexpected response is received from a worker request.
    """
    pass


class WorldAlreadyInitialized(Exception):
    """
    Raised when attempting to initialize the World when it has already been initialized.
    """
    pass


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
            # TODO get real ipm of main instance
            # set to a sentinel value to avoid issues with speed comparisons
            self.avg_ipm = 0
            return

        self.address = address
        self.port = port
        self.verify_remotes = verify_remotes
        self.response_time = None

        if uuid is not None:
            self.uuid = uuid

    def __str__(self):
        return f"{self.address}:{self.port}"

    def info(self, benchmark_payload) -> dict:
        """
         Stores the payload used to benchmark the world and certain attributes of the worker.
         These things are used to draw certain conclusions after the first session.

         Args:
             benchmark_payload (dict): The payload used the benchmark.

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
        Returns the mean absolute percent error using all the currently stored eta percent errors.

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
        return f"https://{self.__str__()}/sdapi/v1/{route}"

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

        eta = self.batch_eta(payload=pseudo_payload)
        return eta

    # TODO separate network latency from total eta error
    def batch_eta(self, payload: dict) -> float:
        """estimate how long it will take to generate <batch_size> images on a worker in seconds"""
        global benchmark_payload
        steps = payload['steps']
        num_images = payload['batch_size']

        # if worker has not yet been benchmarked then
        try:
            eta = (num_images / self.avg_ipm) * 60
            # show effect of increased step size
            real_steps_to_benched = steps / benchmark_payload['steps']
            eta = eta * real_steps_to_benched

            # show effect of high-res fix
            if payload['enable_hr'] is True:
                eta += self.batch_eta_hr(payload=payload)

            # show effect of image size
            real_pix_to_benched = (payload['width'] * payload['height'])\
                / (benchmark_payload['width'] * benchmark_payload['height'])

            eta = eta * real_pix_to_benched
            # show effect of using a sampler other than euler a
            if payload['sampler_name'] != 'Euler a':
                try:
                    percent_difference = self.other_to_euler_a[payload['sampler_name']]
                    if percent_difference > 0:
                        eta = eta - (eta * abs((percent_difference / 100)))
                    else:
                        eta = eta + (eta * abs((percent_difference / 100)))
                except KeyError:
                    print(f"Sampler '{payload['sampler_name']}' efficiency is not recorded.\n")
                    print(f"Sampler efficiency will be treated as equivalent to Euler A.")

            # TODO save and load each workers MPE before the end of session to workers.json.
            #  That way initial estimations are more accurate from the second sdwui session onward
            # adjust for a known inaccuracy in our estimation of this worker using average percent error
            if len(self.eta_percent_error) > 0:
                # eta = eta / (1 - self.eta_mpe() / 100)
                correction = eta * (self.eta_mpe() / 100)
                print(f"worker '{self.uuid}' with correction +{eta + correction} - {eta - correction}")
                if correction > 0:
                    eta = eta + correction
                else:
                    eta = eta - correction

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

        # TODO handle no connection exception and remove worker (for this request) in that case
        # TODO detect remote out of memory exception and restart or garbage collect instance using api?
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
            print(f"Worker '{self.uuid}' {free_vram:.2f}/{total_vram:.2f} GB VRAM free\n")
            self.free_vram = bytes(memory_response['free'])

        if sync_options is True:
            options_response = requests.post(
                self.full_url("options"),
                json=option_payload,
                verify=self.verify_remotes
            )
            self.response = options_response
            # TODO api returns 200 even if it fails to successfully set the checkpoint so we will have to make a
            #  second GET to see if everything loaded...

        if self.benchmarked:
            eta = self.batch_eta(payload=payload)
            print(f"worker '{self.uuid}' predicts it will take {eta:.3f}s to generate {payload['batch_size']} image("
                  f"s) at a speed of {self.avg_ipm} ipm\n")

        try:
            start = time.time()
            response = requests.post(
                self.full_url("txt2img"),
                json=payload,
                verify=self.verify_remotes
            )
            self.response = response.json()

            if self.benchmarked:
                self.response_time = time.time() - start
                variance = ((eta - self.response_time) / self.response_time) * 100
                print(f"\nWorker '{self.uuid}' was off by {variance:.2f}%.\n")
                print(f"Predicted {eta:.2f}s. Actual: {self.response_time:.2f}s\n")

                if self.eta_percent_error == 0:
                    self.eta_percent_error[0] = variance
                else:
                    self.eta_percent_error.append(variance)

        except Exception as e:
            if payload['batch_size'] == 0:
                raise InvalidWorkerResponse("Tried to request a null amount of images")
            else:
                raise InvalidWorkerResponse(e)

        return

    def benchmark(self) -> int:
        """
        given a worker, run a small benchmark and return its performance in images/minute
        makes standard request(s) of 512x512 images and averages them to get the result
        """
        global benchmark_payload

        t: Thread
        samples = 2  # number of times to benchmark the remote / accuracy
        warmup_samples = 2  # number of samples to do before recording as a valid sample in order to "warm-up"

        print(f"Benchmarking worker '{self.uuid}':\n")

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
        for i in range(0, samples + warmup_samples):  # run some extra times so that the remote can "warm up"
            t = Thread(target=self.request, args=(benchmark_payload, None, False,))
            try:  # if the worker is unreachable/offline then handle that here
                t.start()
                start = time.time()
                t.join()
                elapsed = time.time() - start
                sample_ipm = ipm(elapsed)
            # except InvalidWorkerResponse:
            except Exception as e:
                print(e)
                # print(f"Check that worker '{self.uuid}' at {self} is online and port accepting sdwui api requests\n")
                continue

            if i >= warmup_samples:
                print(f"Sample {i - warmup_samples + 1}: Worker '{self.uuid}'({self}) - {sample_ipm:.2f} image(s) per "
                      f"minute\n")
                results.append(sample_ipm)
            elif i == warmup_samples - 1:
                print(f"{self.uuid} warming up\n")

        # average the sample results for accuracy
        ipm_sum = 0
        for ipm in results:
            ipm_sum += ipm
        avg_ipm = math.floor(ipm_sum / samples)

        print(f"Worker '{self.uuid}' average ipm: {avg_ipm}")
        self.avg_ipm = avg_ipm
        # noinspection PyTypeChecker
        self.response = None
        self.benchmarked = True
        return avg_ipm


class Job:
    """
    Keeps track of how much work a given worker should handle.

    Args:
        worker (Worker): The worker to assign the job to.
        batch_size (int): How many images the job, initially, should generate.
    """

    def __init__(self, worker: Worker, batch_size: int):
        self.worker: Worker = worker
        self.batch_size: int = batch_size
        self.complementary: bool = False


class World:
    """
    The frame or "world" which holds all workers (including the local machine).

    Args:
        initial_payload: The original txt2img payload created by the user initiating the generation request on master.
        verify_remotes (bool): Whether to validate remote worker certificates.
    """

    # I'd rather keep the sdwui root directory clean.
    this_extension_path = Path(abspath(getsourcefile(lambda: 0))).parent.parent.parent
    worker_info_path = this_extension_path.joinpath('workers.json')

    def __init__(self, initial_payload, verify_remotes: bool = True):
        master_worker = Worker(master=True)
        self.total_batch_size: int = 0
        self.workers: List[Worker] = [master_worker]
        self.jobs: List[Job] = []
        self.job_timeout: int = 10  # seconds
        self.initialized: bool = False
        self.verify_remotes = verify_remotes
        self.initial_payload = copy.copy(initial_payload)

    def update_world(self, total_batch_size):
        """
        Updates the world with information vital to handling the local generation request after
            the world has already been initialized.

        Args:
            total_batch_size (int): The total number of images requested by the local/master sdwui instance.
        """

        world_size = self.get_world_size()
        if total_batch_size < world_size:
            self.total_batch_size = world_size
            print(f"Total batch size should not be less than the number of workers.\n")
            print(f"Defaulting to a total batch size of '{world_size}' in order to accommodate all workers")
        else:
            self.total_batch_size = total_batch_size

        default_worker_batch_size = self.get_default_worker_batch_size()
        self.sync_master(batch_size=default_worker_batch_size)
        self.update_worker_jobs()
        # self.optimize_jobs(batch_size=default_worker_batch_size)

    def initialize(self, total_batch_size):
        """should be called before a world instance is used for anything"""
        if self.initialized:
            raise WorldAlreadyInitialized("This world instance was already initialized")

        self.benchmark()
        self.update_world(total_batch_size=total_batch_size)
        self.initialized = True

    def get_default_worker_batch_size(self) -> int:
        """the amount of images/total images requested that a worker would compute if conditions were perfect and
        each worker generated at the same speed"""

        return self.total_batch_size // self.get_world_size()

    def get_world_size(self) -> int:
        """
        Returns:
            int: The number of nodes currently registered in the world.
        """
        return len(self.workers)

    def sync_master(self, batch_size: int):
        """
        update the master node's pseudo-job with <batch_size> of images it will be processing
        """

        if len(self.jobs) < 1:
            master_job = Job(worker=self.workers[0], batch_size=batch_size)
            self.jobs.append(master_job)
        else:
            self.master_job().batch_size = batch_size

    def get_master_batch_size(self) -> int:
        """
        Returns:
            int: The number of images the master worker is currently set to generate.
        """
        return self.master_job().batch_size

    def master(self) -> Worker:
        """
        May perform additional checks in the future
        Returns:
            Worker: The local/master worker object.
        """

        return self.workers[0]

    def master_job(self) -> Job:
        """
        May perform additional checks in the future
        Returns:
            Job: The local/master worker job object.
        """

        return self.jobs[0]

    def add_worker(self, uuid: str, address: str, port: int):
        """
        Registers a worker with the world.

        Args:
            uuid (str): The name or unique identifier.
            address (str): The ip or FQDN.
            port (int): The port number.
        """

        worker = Worker(uuid=uuid, address=address, port=port, verify_remotes=self.verify_remotes)
        self.workers.append(worker)

    def benchmark(self):
        """
        Attempts to benchmark all workers a part of the world.
        """

        global benchmark_payload
        workers_info: dict = {}
        saved: bool = os.path.exists(self.worker_info_path)
        benchmark_payload_loaded: bool = False

        if saved:
            workers_info = json.load(open(self.worker_info_path, 'r'))

        # benchmark all nodes
        for worker in self.workers:

            if not saved:
                if worker.master:
                    self.master().avg_ipm = self.benchmark_master()
                    workers_info.update(self.master().info(benchmark_payload=benchmark_payload))
                else:
                    worker.benchmark()
            else:
                if not benchmark_payload_loaded:
                    benchmark_payload = workers_info[worker.uuid]['benchmark_payload']
                    benchmark_payload_loaded = True
                    print("loaded saved worker configuration:")
                    print(workers_info)
                worker.avg_ipm = workers_info[worker.uuid]['avg_ipm']

            workers_info.update(worker.info(benchmark_payload=benchmark_payload))

        json.dump(workers_info, open(self.worker_info_path, 'w'), indent=3)

    def get_current_output_size(self) -> int:
        """
        returns how many images would be returned from all jobs
        """

        num_images = 0

        for job in self.jobs:
            num_images += job.batch_size

        return num_images

    # TODO broken
    def print_speed_stats(self):
        """
        Prints workers by their ipm in descending order.
        """
        workers_copy = copy.deepcopy(self.workers)

        i = 1
        workers_copy.sort(key=lambda w: w.avg_ipm, reverse=True)
        print("Worker speed hierarchy:")
        for worker in workers_copy:
            print(f"{i}.    worker '{worker}' - {worker.avg_ipm} ipm")
            i += 1

    def realtime_jobs(self) -> List[Job]:
        """
        Determines which jobs are considered real-time by checking which jobs are not(complementary).

        Returns:
            fast_jobs (List[Job]): List containing all jobs considered real-time.
        """
        fast_jobs: List[Job] = []

        for job in self.jobs:
            if job.complementary is False:
                fast_jobs.append(job)

        return fast_jobs

    def slowest_realtime_job(self) -> Job:
        """
        Finds the slowest Job that is considered real-time.

        Returns:
            Job: The slowest real-time job.
        """

        return sorted(self.realtime_jobs(), key=lambda job: job.worker.avg_ipm, reverse=False)[0]

    def fastest_realtime_job(self) -> Job:
        """
        Finds the slowest Job that is considered real-time.

        Returns:
            Job: The slowest real-time job.
        """

        return sorted(self.realtime_jobs(), key=lambda job: job.worker.avg_ipm, reverse=True)[0]

    def job_stall(self, worker: Worker, payload: dict) -> float:
        """
            We assume that the passed worker will do an equal portion of the total request.
            Estimate how much time the user would have to wait for the images to show up.
        """

        fastest_worker = self.fastest_realtime_job().worker
        lag = worker.batch_eta(payload=payload) - fastest_worker.batch_eta(payload=payload)

        return lag

    # TODO account for generation "warm-up" lag
    def benchmark_master(self) -> float:
        """
        Benchmarks the local/master worker.

        Returns:
            float: Local worker speed in ipm
        """

        global benchmark_payload
        master_bench_payload = copy.copy(self.initial_payload)

        # TODO fully clean copied payload of anything that might throw off the calculation
        master_bench_payload.batch_size = benchmark_payload['batch_size']
        master_bench_payload.width = benchmark_payload['width']
        master_bench_payload.height = benchmark_payload['height']
        master_bench_payload.steps = benchmark_payload['steps']
        master_bench_payload.prompt = benchmark_payload['prompt']
        master_bench_payload.negative_prompt = benchmark_payload['negative_prompt']
        master_bench_payload.enable_hr = False
        master_bench_payload.disable_extra_networks = True

        # make it seem as though this never happened
        import modules.shared as shared
        state_cache = copy.deepcopy(shared.state)
        start = time.time()
        process_images(master_bench_payload)
        elapsed = time.time() - start
        shared.state = state_cache

        ipm = benchmark_payload['batch_size'] / (elapsed / 60)

        print(f"Master benchmark took {elapsed}: {ipm} ipm")
        self.master().benchmarked = True
        return ipm

    def update_worker_jobs(self):
        """creates initial jobs (before optimization) """
        default_job_size = self.get_default_worker_batch_size()

        # clear jobs if this is not the first time running
        if self.initialized:
            master_job = self.jobs[0]
            self.jobs = [master_job]

        for worker in self.workers:
            if worker.master:
                self.master_job().batch_size = default_job_size
                continue

            batch_size = default_job_size
            self.jobs.append(Job(worker=worker, batch_size=batch_size))

    def optimize_jobs(self, payload: json):
        """
        The payload batch_size should be set to whatever the default worker batch_size would be. 
        get_default_worker_batch_size() should return the proper value if the world is initialized
        Ex. 3 workers(including master): payload['batch_size'] should evaluate to 1
        """

        deferred_images = 0  # the number of images that were not assigned to a worker due to the worker being too slow
        # the maximum amount of images that a "slow" worker can produce in the slack space where other nodes are working
        max_compensation = 4
        images_per_job = None

        for job in self.jobs:

            lag = self.job_stall(job.worker, payload=payload)

            if lag < self.job_timeout:
                job.batch_size = payload['batch_size']
                continue

            print(f"worker '{job.worker.uuid}' would stall the image gallery by ~{lag:.2f}s\n")
            job.complementary = True
            deferred_images = deferred_images + payload['batch_size']
            job.batch_size = 0

        ####################################################
        # redistributing deferred images to realtime jobs  #
        ####################################################

        if deferred_images > 0:
            realtime_jobs = self.realtime_jobs()
            images_per_job = deferred_images // len(realtime_jobs)
            for job in realtime_jobs:
                job.batch_size = job.batch_size + images_per_job

        #####################################
        # complementary worker distribution #
        #####################################

        # Now that this worker would (otherwise) not be doing anything, see if it can still do something.
        # Calculate how many images it can output in the time that it takes the slowest real-time worker to do so.

        for job in self.jobs:
            if job.complementary is False:
                continue

            slowest_active_worker = self.slowest_realtime_job().worker
            slack_time = slowest_active_worker.batch_eta(payload=payload)
            # in the case that this worker is now taking on what others workers would have been (if they were real-time)
            # this means that there will be more slack time for complementary nodes
            slack_time = slack_time + ((slack_time / payload['batch_size']) * images_per_job)

            # see how long it would take to produce only 1 image on this complementary worker
            fake_payload = copy.copy(payload)
            fake_payload['batch_size'] = 1
            secs_per_batch_image = job.worker.batch_eta(payload=fake_payload)
            num_images_compensate = int(slack_time / secs_per_batch_image)

            job.batch_size = num_images_compensate

        # TODO master batch_size cannot be < 1 or it will crash the entire generation.
        #  It might be better to just inject a black image. (if master is that slow)
        master_job = self.master_job()
        if master_job.batch_size < 1:
            master_job.batch_size = 1

        print("After job optimization, job layout is the following:")
        for job in self.jobs:
            print(f"worker '{job.worker.uuid}' - {job.batch_size} images")
        print()
