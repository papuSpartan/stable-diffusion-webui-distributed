import io
import os
import subprocess
from pathlib import Path
import gradio
from scripts.spartan.shared import logger, log_level
from scripts.spartan.Worker import State
from modules.shared import state as webui_state
import json


class UI:
    def __init__(self, script, world):
        self.script = script
        self.world = world

    # handlers
    @staticmethod
    def user_script_btn():
        user_scripts = Path(os.path.abspath(__file__)).parent.parent.joinpath('user')

        for file in user_scripts.iterdir():
            logger.debug(f"found possible script {file.name}")
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

    def benchmark_btn(self):
        logger.info("Redoing benchmarks...")
        self.world.benchmark(rebenchmark=True)

    def clear_queue_btn(self):
        logger.debug(webui_state.__dict__)
        webui_state.end()


    def status_btn(self):
        worker_status = ''
        workers = self.world.get_workers()

        for worker in workers:
            if worker.master:
                continue

            worker_status += f"{worker.uuid} at {worker.address} is {worker.state.name}\n"

        # TODO replace this with a single check to a state flag that we should make in the world class
        for worker in workers:
            if worker.state == State.WORKING:
                return self.world.__str__(), worker_status

        return 'No active jobs!', worker_status

    def save_btn(self, thin_client_mode, job_timeout):
        self.world.thin_client_mode = thin_client_mode
        logger.debug(f"thin client mode is now {thin_client_mode}")
        job_timeout = int(job_timeout)
        self.world.job_timeout = job_timeout
        logger.debug(f"job timeout is now {job_timeout} seconds")

    def save_worker_btn(self, name, address, port, tls):
        worker = self.world.add_worker(name, address, port, tls)

        workers_info = {}
        with open(self.world.worker_info_path, 'r', encoding='utf-8') as worker_info_file:
            try:
                workers_info = json.load(worker_info_file)
            except json.decoder.JSONDecodeError:
                logger.error(f"corrupt or invalid config file... ignoring")
            except io.UnsupportedOperation:
                pass

        with open(self.world.worker_info_path, 'w', encoding='utf-8') as worker_info_file:
            inf: dict = worker.info()
            workers_info[name] = inf[name]

            json.dump(workers_info, worker_info_file, indent=3)


    # end handlers

    def create_root(self):
        with gradio.Box() as root:
            with gradio.Accordion(label='Distributed', open=False):
                with gradio.Tab('Status') as status_tab:
                    status = gradio.Textbox(elem_id='status', show_label=False)
                    status.placeholder = 'Refresh!'
                    jobs = gradio.Textbox(elem_id='jobs', label='Jobs', show_label=True)
                    jobs.placeholder = 'Refresh!'

                    refresh_status_btn = gradio.Button(value='Refresh')
                    refresh_status_btn.style(size='sm')
                    refresh_status_btn.click(self.status_btn, inputs=[], outputs=[jobs, status])

                    status_tab.select(fn=self.status_btn, inputs=[], outputs=[jobs, status])

                with gradio.Tab('Utils'):
                    refresh_checkpoints_btn = gradio.Button(value='Refresh checkpoints')
                    refresh_checkpoints_btn.style(full_width=False)
                    refresh_checkpoints_btn.click(self.world.refresh_checkpoints)

                    run_usr_btn = gradio.Button(value='Run user script')
                    run_usr_btn.style(full_width=False)
                    run_usr_btn.click(self.user_script_btn)

                    interrupt_all_btn = gradio.Button(value='Interrupt all', variant='stop')
                    interrupt_all_btn.style(full_width=False)
                    interrupt_all_btn.click(self.world.interrupt_remotes)

                    redo_benchmarks_btn = gradio.Button(value='Redo benchmarks', variant='stop')
                    redo_benchmarks_btn.style(full_width=False)
                    redo_benchmarks_btn.click(self.benchmark_btn, inputs=[], outputs=[])

                    reload_config_btn = gradio.Button(value='Reload config from file')
                    reload_config_btn.style(full_width=False)
                    reload_config_btn.click(self.world.load_config)

                    if log_level == 'DEBUG':
                        clear_queue_btn = gradio.Button(value='Clear local webui queue', variant='stop')
                        clear_queue_btn.style(full_width=False)
                        clear_queue_btn.click(self.clear_queue_btn)

                with gradio.Tab('Worker Config'):
                    worker_name_field = gradio.Textbox(label='Name')
                    worker_address_field = gradio.Textbox(label='Address')
                    worker_port_field = gradio.Textbox(label='Port', value='7860')
                    worker_tls_cbx = gradio.Checkbox(
                        label='connect to worker using https'
                    )
                    save_worker_btn = gradio.Button(
                        value='Add Worker'
                    )
                    save_worker_btn.click(self.save_worker_btn, inputs=[worker_name_field, worker_address_field, worker_port_field, worker_tls_cbx])

                with gradio.Tab('Settings'):
                    thin_client_cbx = gradio.Checkbox(
                        label='Thin-client mode (experimental)',
                        info="Only generate images using remote workers. There will be no previews when enabled.",
                        value=self.world.thin_client_mode
                    )
                    job_timeout = gradio.Number(
                        label='Job timeout', value=self.world.job_timeout,
                        info="Seconds until a worker is considered too slow to be assigned an"
                             " equal share of the total request. Longer than 2 seconds is recommended."
                    )

                    save_btn = gradio.Button(value='Update')
                    save_btn.click(fn=self.save_btn, inputs=[thin_client_cbx, job_timeout])

                with gradio.Tab('Help'):
                    gradio.Markdown(
                        """
                        - [Discord Server 🤝](https://discord.gg/Jpc8wnftd4)
                        - [Github Repository </>](https://github.com/papuSpartan/stable-diffusion-webui-distributed)
                        """
                    )

            return root
