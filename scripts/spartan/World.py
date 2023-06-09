"""
This module facilitates the creation of a stable-diffusion-webui centered distributed computing system.

World:
    The main class which should be instantiated in order to create a new sdwui distributed system.
"""

import copy
import json
import os
import time
from typing import List
from threading import Thread
from inspect import getsourcefile
from os.path import abspath
from pathlib import Path
from modules.processing import process_images, StableDiffusionProcessingTxt2Img
import modules.shared as shared
from scripts.spartan.Worker import Worker
from scripts.spartan.shared import logger, warmup_samples, benchmark_payload


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
        suffix = f"Job: {self.batch_size} images. Owned by '{self.worker.uuid}'. Rate: {self.worker.avg_ipm}ipm"
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
    this_extension_path = Path(abspath(getsourcefile(lambda: 0))).parent.parent.parent
    worker_info_path = this_extension_path.joinpath('workers.json')

    def __init__(self, initial_payload, verify_remotes: bool = True):
        master_worker = Worker(master=True)
        self.total_batch_size: int = 0
        self.workers: List[Worker] = [master_worker]
        self.jobs: List[Job] = []
        self.job_timeout: int = 6  # seconds
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

        self.total_batch_size = total_batch_size

        default_worker_batch_size = self.get_default_worker_batch_size()
        self.sync_master(batch_size=default_worker_batch_size)
        self.update_worker_jobs()

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

    def interrupt_remotes(self):

        for worker in self.workers:
            if worker.master:
                continue

            t = Thread(target=worker.interrupt, args=())
            t.start()

    def refresh_checkpoints(self):
        for worker in self.workers:
            if worker.master:
                continue

            t = Thread(target=worker.refresh_checkpoints, args=())
            t.start()

    def benchmark(self, rebenchmark: bool = False):
        """
        Attempts to benchmark all workers a part of the world.
        """
        global benchmark_payload

        workers_info: dict = {}
        saved: bool = os.path.exists(self.worker_info_path)
        unbenched_workers = []
        benchmark_threads = []

        def benchmark_wrapped(worker):
            bench_func = worker.benchmark if not worker.master else self.benchmark_master
            worker.avg_ipm = bench_func()
            worker.benchmarked = True

        if rebenchmark:
            saved = False
            for worker in self.workers:
                worker.benchmarked = False
            unbenched_workers = self.workers

        if saved:
            with open(self.worker_info_path, 'r') as worker_info_file:
                try:
                    workers_info = json.load(worker_info_file)
                except json.JSONDecodeError:
                    logger.error(f"workers.json is not valid JSON, regenerating")
                    rebenchmark = True
                    unbenched_workers = self.workers

        # load stats for any workers that have already been benched
        if saved and not rebenchmark:
            logger.debug(f"loaded saved configuration: \n{workers_info}")

            for worker in self.workers:
                try:
                    worker.avg_ipm = workers_info[worker.uuid]['avg_ipm']
                    worker.benchmarked = True
                except KeyError:
                    logger.debug(f"worker '{worker.uuid}' not found in workers.json")
                    unbenched_workers.append(worker)
            return
        else:
            unbenched_workers = self.workers

        # benchmark those that haven't been
        for worker in unbenched_workers:
            t = Thread(target=benchmark_wrapped, args=(worker, ), name=f"{worker.uuid}_benchmark")
            benchmark_threads.append(t)
            t.start()
            logger.info(f"benchmarking worker '{worker.uuid}'")

        # wait for all benchmarks to finish and update stats on newly benchmarked workers
        if len(benchmark_threads) > 0:
            with open(self.worker_info_path, 'w') as worker_info_file:
                for t in benchmark_threads:
                    t.join()
                logger.info("Benchmarking finished")

                for worker in unbenched_workers:
                    workers_info.update(worker.info())
                workers_info.update({'benchmark_payload': benchmark_payload})

                # save benchmark results to workers.json
                json.dump(workers_info, worker_info_file, indent=3)

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
        workers_copy = copy.deepcopy(self.workers)
        workers_copy.sort(key=lambda w: w.avg_ipm, reverse=True)

        total_ipm = 0
        for worker in workers_copy:
            total_ipm += worker.avg_ipm

        i = 1
        output = "World composition:\n"
        for worker in workers_copy:
            output += f"{i}. '{worker.uuid}'({worker}) - {worker.avg_ipm:.2f} ipm\n"
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
        for key in benchmark_payload:
            setattr(master_bench_payload, key, benchmark_payload[key])
        # Keeps from trying to save the images when we don't know the path. Also, there's not really any reason to.
        master_bench_payload.do_not_save_samples = True

        # "warm up" due to initial generation lag
        for i in range(warmup_samples):
            process_images(master_bench_payload)

        # get actual sample
        start = time.time()
        process_images(master_bench_payload)
        elapsed = time.time() - start

        ipm = benchmark_payload['batch_size'] / (elapsed / 60)

        logger.debug(f"Master benchmark took {elapsed:.2f}: {ipm:.2f} ipm")
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
        # max_compensation = 4 currently unused
        images_per_job = None
        images_checked = 0
        for job in self.jobs:

            lag = self.job_stall(job.worker, payload=payload)

            if lag < self.job_timeout or lag == 0:
                job.batch_size = payload['batch_size']
                images_checked += payload['batch_size']
                continue

            logger.debug(f"worker '{job.worker.uuid}' would stall the image gallery by ~{lag:.2f}s\n")
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
            logger.debug(f"There's {slack_time:.2f}s of slack time available for worker '{job.worker.uuid}'")

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

        # TODO master batch_size cannot be < 1 or it will crash the entire generation.
        #  It might be better to just inject a black image. (if master is that slow)
        master_job = self.master_job()
        if master_job.batch_size < 1:
            logger.warning("Master couldn't keep up... defaulting to 1 image")
            master_job.batch_size = 1



        logger.info("Job distribution:")
        iterations = payload['n_iter']
        logger.info(f"{self.total_batch_size} * {iterations} iteration(s): {self.total_batch_size * iterations} images")
        for job in self.jobs:
            logger.info(f"worker '{job.worker.uuid}' - {job.batch_size * iterations} images")
        print()
