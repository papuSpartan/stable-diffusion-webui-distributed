"""
This module facilitates the creation of a stable-diffusion-webui centered distributed computing system.

World:
    The main class which should be instantiated in order to create a new sdwui distributed system.
"""

import copy
import json
import os
import time
from typing import List, Dict, Union
from threading import Thread
from modules.processing import process_images, StableDiffusionProcessingTxt2Img
import modules.shared as shared
from .Worker import Worker, State
from .shared import logger, warmup_samples, extension_path
from .pmodels import Config_Model, Benchmark_Payload
from . import shared as sh


class NotBenchmarked(Exception):
    """
    Should be raised when attempting to do something that requires knowledge of worker benchmark statistics, and
    they haven't been calculated yet.
    """
    pass


class WorldAlreadyInitialized(Exception):
    """
    Raised when attempting to initialize the World when it has already been initialized.
    """
    pass


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

    def __str__(self):
        prefix = ''
        suffix = f"Job: {self.batch_size} image(s) owned by '{self.worker.label}'. Rate: {self.worker.avg_ipm:0.2f} ipm"
        if self.complementary:
            prefix = "(complementary) "

        return prefix + suffix


class World:
    """
    The frame or "world" which holds all workers (including the local machine).

    Args:
        initial_payload: The original txt2img payload created by the user initiating the generation request on master.
        verify_remotes (bool): Whether to validate remote worker certificates.
    """

    # I'd rather keep the sdwui root directory clean.
    config_path = shared.cmd_opts.distributed_config
    old_config_path = worker_info_path = extension_path.joinpath('workers.json')

    def __init__(self, initial_payload, verify_remotes: bool = True):
        self.master_worker = Worker(master=True)
        self.total_batch_size: int = 0
        self._workers: List[Worker] = [self.master_worker]
        self.jobs: List[Job] = []
        self.job_timeout: int = 3  # seconds
        self.initialized: bool = False
        self.verify_remotes = verify_remotes
        self.initial_payload = copy.copy(initial_payload)
        self.thin_client_mode = False

    def __getitem__(self, label: str) -> Worker:
        for worker in self._workers:
            if worker.label == label:
                return worker

    def __repr__(self):
        return f"{len(self._workers)} workers"

    def update_world(self, total_batch_size):
        """
        Updates the world with information vital to handling the local generation request after
            the world has already been initialized.

        Args:
            total_batch_size (int): The total number of images requested by the local/master sdwui instance.
        """

        self.total_batch_size = total_batch_size
        self.update_jobs()

    def initialize(self, total_batch_size):
        """should be called before a world instance is used for anything"""
        if self.initialized:
            raise WorldAlreadyInitialized("This world instance was already initialized")

        self.benchmark()
        self.update_world(total_batch_size=total_batch_size)
        self.initialized = True

    def default_batch_size(self) -> int:
        """the amount of images/total images requested that a worker would compute if conditions were perfect and
        each worker generated at the same speed. assumes one batch only"""

        return self.total_batch_size // self.size()

    def size(self) -> int:
        """
        Returns:
            int: The number of nodes currently registered in the world.
        """
        return len(self.get_workers())

    def master(self) -> Worker:
        """
        May perform additional checks in the future
        Returns:
            Worker: The local/master worker object.
        """

        return self.master_worker

    def master_job(self) -> Job:
        """
        May perform additional checks in the future
        Returns:
            Job: The local/master worker job object.
        """

        for job in self.jobs:
            if job.worker.master:
                return job

        raise Exception("Master job not found")

    def add_worker(self, **kwargs):
        """
        Registers a worker with the world.

        Returns:
            Worker: The worker object.
        """

        original = self[kwargs['label']]  # if worker doesn't already exist then just make a new one
        if original is None:
            new = Worker(**kwargs)
            self._workers.append(Worker(**kwargs))
            return new
        else:
            for key in kwargs:
                if hasattr(original, key):
                    # TODO only necessary because this is skipping Worker.__init__ and the pyd model is saving the state as an int instead of an actual enum
                    if key == 'state':
                        original.state = kwargs[key] if type(kwargs[key]) is State else State(kwargs[key])
                        continue

                    setattr(original, key, kwargs[key])

            return original

    def interrupt_remotes(self):

        for worker in self.get_workers():
            if worker.master:
                continue

            t = Thread(target=worker.interrupt, args=())
            t.start()

    def refresh_checkpoints(self):
        for worker in self.get_workers():
            if worker.master:
                continue

            t = Thread(target=worker.refresh_checkpoints, args=())
            t.start()

    def benchmark(self, rebenchmark: bool = False):
        """
        Attempts to benchmark all workers a part of the world.
        """

        unbenched_workers = []
        benchmark_threads = []

        def benchmark_wrapped(worker):
            bench_func = worker.benchmark if not worker.master else self.benchmark_master
            worker.avg_ipm = bench_func()
            worker.benchmarked = True

        if rebenchmark:
            for worker in self._workers:
                worker.benchmarked = False
            unbenched_workers = self._workers
        else:
            self.load_config()

            for worker in self._workers:
                if worker.avg_ipm is None or worker.avg_ipm <= 0:
                    logger.debug(f"recorded speed for worker '{worker.label}' is invalid")
                    unbenched_workers.append(worker)
                else:
                    worker.benchmarked = True

        # benchmark those that haven't been
        for worker in unbenched_workers:
            t = Thread(target=benchmark_wrapped, args=(worker, ), name=f"{worker.label}_benchmark")
            benchmark_threads.append(t)
            t.start()
            logger.info(f"benchmarking worker '{worker.label}'")

        # wait for all benchmarks to finish and update stats on newly benchmarked workers
        if len(benchmark_threads) > 0:
            for t in benchmark_threads:
                t.join()
            logger.info("benchmarking finished")

            # save benchmark results to workers.json
            self.save_config()
            logger.info(self.speed_summary())

    def get_current_output_size(self) -> int:
        """
        returns how many images would be returned from all jobs
        """

        num_images = 0

        for job in self.jobs:
            num_images += job.batch_size

        return num_images

    def speed_summary(self) -> str:
        """
        Returns string listing workers by their ipm in descending order.
        """
        workers_copy = copy.deepcopy(self._workers)
        workers_copy.sort(key=lambda w: w.avg_ipm, reverse=True)

        total_ipm = 0
        for worker in workers_copy:
            total_ipm += worker.avg_ipm

        i = 1
        output = "World composition:\n"
        for worker in workers_copy:
            output += f"{i}. '{worker.label}'({worker}) - {worker.avg_ipm:.2f} ipm\n"
            i += 1
        output += f"total: ~{total_ipm:.2f} ipm"

        return output

    def __str__(self):
        # print status of all jobs
        jobs_str = ""
        for job in self.jobs:
            jobs_str += job.__str__() + "\n"

        return jobs_str

    def realtime_jobs(self) -> List[Job]:
        """
        Determines which jobs are considered real-time by checking which jobs are not(complementary).

        Returns:
            fast_jobs (List[Job]): List containing all jobs considered real-time.
        """
        fast_jobs: List[Job] = []

        for job in self.jobs:
            if job.worker.benchmarked is False or job.worker.avg_ipm is None:
                continue

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
        # if the worker is the fastest, then there is no lag
        if worker == fastest_worker:
            return 0

        lag = worker.batch_eta(payload=payload, quiet=True) - fastest_worker.batch_eta(payload=payload, quiet=True)

        return lag

    def benchmark_master(self) -> float:
        """
        Benchmarks the local/master worker.

        Returns:
            float: Local worker speed in ipm
        """

        # wrap our benchmark payload
        master_bench_payload = StableDiffusionProcessingTxt2Img()
        d = sh.benchmark_payload.dict()
        for key in d:
            setattr(master_bench_payload, key, d[key])

        # Keeps from trying to save the images when we don't know the path. Also, there's not really any reason to.
        master_bench_payload.do_not_save_samples = True

        # "warm up" due to initial generation lag
        for i in range(warmup_samples):
            process_images(master_bench_payload)

        # get actual sample
        start = time.time()
        process_images(master_bench_payload)
        elapsed = time.time() - start

        ipm = sh.benchmark_payload.batch_size / (elapsed / 60)

        logger.debug(f"Master benchmark took {elapsed:.2f}: {ipm:.2f} ipm")
        self.master().benchmarked = True
        return ipm

    def update_jobs(self):
        """creates initial jobs (before optimization) """

        # clear jobs if this is not the first time running
        self.jobs = []

        batch_size = self.default_batch_size()
        for worker in self.get_workers():
            if worker.state != State.DISABLED and worker.state != State.UNAVAILABLE:
                if worker.avg_ipm is None or worker.avg_ipm <= 0:
                    logger.debug(f"No recorded speed for worker '{worker.label}, benchmarking'")
                    worker.benchmark()

                self.jobs.append(Job(worker=worker, batch_size=batch_size))

    def get_workers(self):
        filtered: List[Worker] = []
        for worker in self._workers:
            if worker.avg_ipm is not None and worker.avg_ipm <= 0:
                logger.warning(f"config reports invalid speed (0 ipm) for worker '{worker.label}', setting default of 1 ipm.\nplease re-benchmark")
                worker.avg_ipm = 1
                continue
            if worker.master and self.thin_client_mode:
                continue
            if worker.state != State.UNAVAILABLE and worker.state != State.DISABLED:
                filtered.append(worker)

        return filtered

    def optimize_jobs(self, payload: json):
        """
        The payload batch_size should be set to whatever the default worker batch_size would be. 
        default_batch_size() should return the proper value if the world is initialized
        Ex. 3 workers(including master): payload['batch_size'] should evaluate to 1
        """

        deferred_images = 0  # the number of images that were not assigned to a worker due to the worker being too slow
        # the maximum amount of images that a "slow" worker can produce in the slack space where other nodes are working
        # max_compensation = 4 currently unused
        images_per_job = None
        images_checked = 0
        for job in self.jobs:

            lag = self.job_stall(job.worker, payload=payload)

            if lag < self.job_timeout or lag == 0:
                job.batch_size = payload['batch_size']
                images_checked += payload['batch_size']
                continue

            logger.debug(f"worker '{job.worker.label}' would stall the image gallery by ~{lag:.2f}s\n")
            job.complementary = True
            if deferred_images + images_checked + payload['batch_size'] > self.total_batch_size:
                logger.debug(f"would go over actual requested size")
            else:
                deferred_images += payload['batch_size']
            job.batch_size = 0

        ####################################################
        # redistributing deferred images to realtime jobs  #
        ####################################################

        if deferred_images > 0:
            realtime_jobs = self.realtime_jobs()
            images_per_job = deferred_images // len(realtime_jobs)
            for job in realtime_jobs:
                job.batch_size = job.batch_size + images_per_job

        #######################
        # remainder handling  #
        #######################

        # when total number of requested images was not cleanly divisible by world size then we tack the remainder on
        remainder_images = self.total_batch_size - self.get_current_output_size()
        if remainder_images >= 1:
            logger.debug(f"The requested number of images({self.total_batch_size}) was not cleanly divisible by the number of realtime nodes({len(self.realtime_jobs())}) resulting in {remainder_images} that will be redistributed")

            realtime_jobs = self.realtime_jobs()
            realtime_jobs.sort(key=lambda x: x.batch_size)
            # round-robin distribute the remaining images
            while remainder_images >= 1:
                for job in realtime_jobs:
                    if remainder_images < 1:
                        break
                    job.batch_size += 1
                    remainder_images -= 1

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
            logger.debug(f"There's {slack_time:.2f}s of slack time available for worker '{job.worker.label}'")

            # in the case that this worker is now taking on what others workers would have been (if they were real-time)
            # this means that there will be more slack time for complementary nodes
            if images_per_job is not None:
                slack_time = slack_time + ((slack_time / payload['batch_size']) * images_per_job)

            # see how long it would take to produce only 1 image on this complementary worker
            fake_payload = copy.copy(payload)
            fake_payload['batch_size'] = 1
            secs_per_batch_image = job.worker.batch_eta(payload=fake_payload)
            num_images_compensate = int(slack_time / secs_per_batch_image)

            job.batch_size = num_images_compensate

        distro_summary = "Job distribution:\n"
        iterations = payload['n_iter']
        distro_summary += f"{self.total_batch_size} * {iterations} iteration(s): {self.total_batch_size * iterations} images total\n"
        for job in self.jobs:
            distro_summary += f"'{job.worker.label}' - {job.batch_size * iterations} image(s) @ {job.worker.avg_ipm:.2f} ipm\n"
        logger.info(distro_summary)

        # delete any jobs that have no work
        last = len(self.jobs) - 1
        while last > 0:
            if self.jobs[last].batch_size < 1:
                del self.jobs[last]
            last -= 1

    def config(self) -> dict:
        """
         {
             "workers": [
                 {
                     "worker1": {
                         "address": "<http://www.example.com>"
                     }
                 }, ...
        }
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Config was not found at '{self.config_path}'")
            if os.path.exists(self.old_config_path):

                with open(self.old_config_path) as config_file:
                    old_config = json.load(config_file)
                    config = {"workers": [], "benchmark_payload": old_config.get("benchmark_payload", None)}

                    try:
                        del old_config["benchmark_payload"]
                    except KeyError:
                        pass

                    for worker_label in old_config:
                        fields = old_config[worker_label]
                        fields["address"] = "localhost"  # this should be overwritten by add_worker() getting the address from --distributed-remotes
                        config["workers"].append({worker_label: fields})

                    logger.info(f"translated legacy config")
                    return config
            else:
                open(self.config_path, 'w')
                logger.info(f"Generated new config file at '{self.config_path}'")

        with open(self.config_path, 'r') as config:
            try:
                return json.load(config)
            except json.decoder.JSONDecodeError:
                logger.error(f"config is corrupt or invalid JSON, unable to load")

    def load_config(self):
        """
        Loads the config file and adds workers to the world.
        This function should be called after worker command arguments are parsed.
        """
        config_raw = self.config()
        if config_raw is None:
            logger.debug("cannot parse null config (present but empty config file?)")
            sh.benchmark_payload = Benchmark_Payload()
            return

        config = Config_Model(**config_raw)

        # saves config schema to <extension>/distributed-config.schema.json
        # print(models.Config.schema_json())
        # with open(self.extension_path.joinpath("distributed-config.schema.json"), "w") as schema_file:
        #     json.dump(json.loads(models.Config.schema_json()), schema_file, indent=3)

        for w in config.workers:
            label = next(iter(w.keys()))
            fields = w[label].__dict__
            fields['label'] = label

            self.add_worker(**fields)

        sh.benchmark_payload = Benchmark_Payload(**config.benchmark_payload)
        self.job_timeout = config.job_timeout

        logger.debug("config loaded")

    def save_config(self):
        """
        Saves the config file.
        """

        config = Config_Model(
            workers=[{worker.label: worker.model.dict()} for worker in self._workers],
            benchmark_payload=sh.benchmark_payload,
            job_timeout=self.job_timeout
        )

        with open(self.config_path, 'w+') as config_file:
            config_file.write(config.json(indent=3))
            logger.debug(f"config saved")

    def ping_remotes(self, indiscriminate: bool = False):
        """
        Checks to see which workers are reachable over the network and marks those that are not as such

        Args:
            indiscriminate: if True, also pings workers thought to already be reachable (State.IDLE)
        """
        for worker in self._workers:
            if worker.master:
                continue
            if worker.state == State.DISABLED:
                logger.debug(f"refusing to ping disabled worker '{worker.label}'")
                continue

            if worker.state == State.UNAVAILABLE or indiscriminate is True:
                logger.debug(f"checking if worker '{worker.label}' is reachable...")
                reachable = worker.reachable()
                if reachable:
                    if worker.queried and worker.state == State.IDLE:  # TODO worker.queried
                        continue

                    worker.supported_scripts = worker.session.get(url=worker.full_url('scripts')).json()
                    logger.info(f"worker '{worker.label}' is online, marking as available")
                    worker.state = State.IDLE
                else:
                    logger.info(f"worker '{worker.label}' is unreachable")
