import os
import subprocess
from pathlib import Path
from threading import Thread
import gradio
from modules.shared import opts
from modules.shared import state as webui_state
from .shared import logger, LOG_LEVEL, gui_handler
from .worker import State
from modules.call_queue import queue_lock
from modules import progress
from modules.ui_components import InputAccordion

worker_select_dropdown = None

class UI:
    """extension user interface related things"""

    def __init__(self, world, is_img2img):
        self.world = world
        self.original_model_dropdown_handler = opts.data_labels.get('sd_model_checkpoint').onchange
        self.is_img2img = is_img2img

    # handlers
    @staticmethod
    def user_script_btn():
        """executes a script placed by the user at <extension>/user/sync*"""
        user_scripts = Path(os.path.abspath(__file__)).parent.parent.joinpath('user')

        user_script = None
        for file in user_scripts.iterdir():
            logger.debug(f"found possible script {file.name}")
            if file.is_file() and file.name.startswith('sync'):
                user_script = file
        if user_script is None:
            logger.error(
                "couldn't find user script\n"
                "script must be placed under <extension>/user/ and filename must begin with sync"
            )
            return False

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
        """benchmarks all registered workers that aren't unavailable"""
        logger.info("Redoing benchmarks...")
        self.world.benchmark(rebenchmark=True)

    @staticmethod
    def clear_queue_btn():
        """debug utility that will clear the internal webui queue. sometimes good for jams"""
        logger.debug(webui_state.__dict__)
        webui_state.end()
        progress.pending_tasks.clear()
        progress.current_task = None
        if queue_lock._lock.locked():
            queue_lock.release()

    def status_btn(self):
        """updates a simplified overview of registered workers and their jobs"""
        worker_status = ''
        workers = self.world._workers
        logs = gui_handler.dump()

        for worker in workers:
            if worker.master:
                continue

            worker_status += f"{worker.label} at {worker.address} is {worker.state.name}\n"

        for worker in workers:
            if worker.state == State.WORKING:
                return str(self.world), worker_status, logs

        return 'No active jobs!', worker_status, logs

    def save_btn(self, thin_client_mode, job_timeout, complement_production, step_scaling):
        """updates the options visible on the settings tab"""

        self.world.thin_client_mode = thin_client_mode
        job_timeout = int(job_timeout)
        self.world.job_timeout = job_timeout
        self.world.complement_production = complement_production
        self.world.step_scaling = step_scaling
        self.world.save_config()

    def save_worker_btn(self, label, address, port, tls, disabled):
        """creates or updates the worker selected in the worker config tab"""

        # determine what state to save
        # if updating a pre-existing worker then grab its current state
        if not label == 'master':
            state = State.IDLE
            if disabled:
                state = State.DISABLED
            else:
                original = self.world[label]
                if original is not None:
                    state = original.state if original.state != State.DISABLED else State.IDLE

            self.world.add_worker(
                label=label,
                address=address,
                port=port if len(port) > 0 else 7860,
                tls=tls,
                state=state
            )
            self.world.save_config()

        # visibly update which workers can be selected
        labels = sorted([worker.label for worker in self.world._workers])
        return gradio.Dropdown.update(choices=labels)

    def remove_worker_btn(self, worker_label):
        """removes, from disk and memory, whatever worker is selected on the worker config tab"""

        if not worker_label == 'master':
            # remove worker from memory
            for worker in self.world._workers:
                if worker.label == worker_label:
                    self.world._workers.remove(worker)

            # remove worker from disk
            self.world.save_config()

        # visibly update which workers can be selected
        labels = sorted([worker.label for worker in self.world._workers])
        return gradio.Dropdown.update(choices=labels)

    def populate_worker_config_from_selection(self, selection):
        """populates the ui components on the worker config tab with the current values of the selected worker"""
        selected_worker = self.world[selection]
        available_models = selected_worker.available_models()
        if not selected_worker.master:
            if len(available_models) > 0:
                available_models.append('None')  # for disabling override
        else:
            available_models.append('N/A')

        return [
            gradio.Textbox.update(value=selected_worker.address),
            gradio.Textbox.update(value=selected_worker.port),
            gradio.Checkbox.update(value=selected_worker.tls),
            gradio.Dropdown.update(choices=available_models),
            gradio.Checkbox.update(value=selected_worker.state == State.DISABLED)
        ]

    def override_worker_model(self, model, worker_label):
        """forces a worker to always use the selected model in future requests"""
        worker = self.world[worker_label]

        if model == "None":
            worker.model_override = None
            model = opts.sd_model_checkpoint
        else:
            worker.model_override = model

        Thread(target=worker.load_options, args=(model,)).start()

    def update_credentials_btn(self, api_auth_toggle, user, password, worker_label):
        worker = self.world[worker_label]
        if worker.master:
            return

        if api_auth_toggle is False:
            worker.user = None
            worker.password = None
            worker.session.auth = None
        else:
            worker.user = user
            worker.password = password
            worker.session.auth = (user, password)
        self.world.save_config()

    def main_toggle_btn(self, state):
        if self.is_img2img:
            if self.world.enabled_i2i == state: # just prevents a redundant config save if ui desyncs
                return
            self.world.enabled_i2i = state
        else:
            if self.world.enabled == state:
                return
            self.world.enabled = state

        self.world.save_config()

        # restore vanilla sdwui handler for model dropdown if extension is disabled or inject if otherwise
        if not self.world.enabled and not self.world.enabled_i2i:
            model_dropdown = opts.data_labels.get('sd_model_checkpoint')
            if self.original_model_dropdown_handler is not None:
                model_dropdown.onchange = self.original_model_dropdown_handler
            self.world.is_dropdown_handler_injected = False
        else:
            self.world.inject_model_dropdown_handler()

    def reset_error_correction_btn(self):
        for worker in self.world._workers:
            logger.debug(f"Worker '{worker.label}' mpe before wiping:\n{worker.eta_percent_error}")
            worker.eta_percent_error = []
        self.world.save_config()

    # end handlers

    def create_ui(self):
        """creates the extension UI and returns relevant components"""
        components = []
        elem_id = 'enabled'
        if self.is_img2img:
            elem_id += '_i2i'

        with gradio.Blocks(variant='compact'):  # Group() and Box() remove spacing
            with InputAccordion(label='Distributed', open=False, value=self.world.config().get(elem_id), elem_id=elem_id) as main_toggle:
                main_toggle.input(self.main_toggle_btn, inputs=[main_toggle])
                setattr(main_toggle.accordion, 'do_not_save_to_config', True) # InputAccordion is really a CheckBox
                components.append(main_toggle)

                with gradio.Tab('Status') as status_tab:
                    status = gradio.Textbox(elem_id='status', show_label=False, interactive=False)
                    status.placeholder = 'Refresh!'
                    jobs = gradio.Textbox(elem_id='jobs', label='Jobs', show_label=True, interactive=False)
                    jobs.placeholder = 'Refresh!'

                    logs = gradio.Textbox(
                        elem_id='logs',
                        label='Log',
                        show_label=True,
                        interactive=False,
                        max_lines=4,
                        info='top-most message is newest'
                    )

                    refresh_status_btn = gradio.Button(value='Refresh 🔄', size='sm', elem_id='distributed-refresh-status', visible=False)
                    refresh_status_btn.click(self.status_btn, inputs=[], outputs=[jobs, status, logs], show_progress=False)

                    status_tab.select(fn=self.status_btn, inputs=[], outputs=[jobs, status, logs])
                    components += [status, jobs, logs, refresh_status_btn]

                with gradio.Tab('Utils'):
                    with gradio.Row():
                        refresh_checkpoints_btn = gradio.Button(value='🆕 Refresh checkpoints')
                        refresh_checkpoints_btn.click(self.world.refresh_checkpoints)

                        reload_config_btn = gradio.Button(value='📜 Reload config')
                        reload_config_btn.click(self.world.load_config)

                        redo_benchmarks_btn = gradio.Button(value='📊 Redo benchmarks')
                        redo_benchmarks_btn.click(self.benchmark_btn, inputs=[], outputs=[])

                        run_usr_btn = gradio.Button(value='⚙️ Run script')
                        run_usr_btn.click(self.user_script_btn)

                        components += [refresh_checkpoints_btn, run_usr_btn, reload_config_btn, redo_benchmarks_btn]

                    with gradio.Row():
                        reconnect_lost_workers_btn = gradio.Button(value='🔌 Reconnect workers')
                        reconnect_lost_workers_btn.click(self.world.ping_remotes)

                        interrupt_all_btn = gradio.Button(value='⏸️ Interrupt all')
                        interrupt_all_btn.click(self.world.interrupt_remotes)

                        restart_workers_btn = gradio.Button(value="🔁 Restart All")
                        restart_workers_btn.click(
                            _js="confirm_restart_workers",
                            fn=lambda confirmed: self.world.restart_all() if confirmed else None,
                            inputs=[restart_workers_btn],
                            outputs=[]
                        )

                    if LOG_LEVEL == 'DEBUG':
                        clear_queue_btn = gradio.Button(value='Clear local webui queue', variant='stop')
                        clear_queue_btn.click(self.clear_queue_btn)
                        reset_error_correction_btn = gradio.Button(value='Clear ETA MPE')
                        reset_error_correction_btn.click(self.reset_error_correction_btn)
                        components += [clear_queue_btn, reset_error_correction_btn]

                    components += [interrupt_all_btn, redo_benchmarks_btn, restart_workers_btn, reconnect_lost_workers_btn]

                with gradio.Tab('Worker Config'):
                    worker_select_dropdown = gradio.Dropdown(
                        sorted([worker.label for worker in self.world._workers]),
                        info='Select a pre-existing worker or enter a label for a new one',
                        label='Label',
                        allow_custom_value=True
                    )
                    worker_address_field = gradio.Textbox(label='Address', placeholder='localhost')
                    worker_port_field = gradio.Textbox(label='Port', placeholder='7860')
                    worker_tls_cbx = gradio.Checkbox(
                        label='connect using https'
                    )
                    worker_disabled_cbx = gradio.Checkbox(
                        label='disabled'
                    )
                    components += [worker_select_dropdown, worker_address_field, worker_port_field, worker_tls_cbx, worker_disabled_cbx]

                    with gradio.Accordion(label='Advanced', open=False):
                        model_override_dropdown = gradio.Dropdown(label='Model override')
                        model_override_dropdown.select(self.override_worker_model,
                                                       inputs=[model_override_dropdown, worker_select_dropdown])

                        def pixel_cap_handler(selected_worker, pixel_cap):
                            selected = self.world[selected_worker]
                            selected.pixel_cap = pixel_cap
                            self.world.save_config()

                        pixel_cap = gradio.Number(label='Pixel cap')
                        pixel_cap.input(pixel_cap_handler, inputs=[worker_select_dropdown, pixel_cap])

                        # API authentication
                        worker_api_auth_cbx = gradio.Checkbox(label='API Authentication')
                        worker_user_field = gradio.Textbox(label='Username')
                        worker_password_field = gradio.Textbox(label='Password', type='password')
                        update_credentials_btn = gradio.Button(value='Update API Credentials')
                        update_credentials_btn.click(self.update_credentials_btn, inputs=[
                            worker_api_auth_cbx,
                            worker_user_field,
                            worker_password_field,
                            worker_select_dropdown
                        ])

                        components += [model_override_dropdown, pixel_cap, worker_api_auth_cbx, worker_user_field, worker_password_field, update_credentials_btn]

                    with gradio.Row():
                        save_worker_btn = gradio.Button(value='Add/Update Worker')
                        save_worker_btn.click(self.save_worker_btn,
                                              inputs=[worker_select_dropdown,
                                                      worker_address_field,
                                                      worker_port_field,
                                                      worker_tls_cbx,
                                                      worker_disabled_cbx
                                                      ],
                                              outputs=[worker_select_dropdown]
                                              )
                        remove_worker_btn = gradio.Button(value='Remove Worker', variant='stop')
                        remove_worker_btn.click(self.remove_worker_btn, inputs=worker_select_dropdown,
                                                outputs=[worker_select_dropdown])
                        components += [save_worker_btn, remove_worker_btn]

                    worker_select_dropdown.select(
                        self.populate_worker_config_from_selection,
                        inputs=worker_select_dropdown,
                        outputs=[
                            worker_address_field,
                            worker_port_field,
                            worker_tls_cbx,
                            model_override_dropdown,
                            worker_disabled_cbx
                        ]
                    )

                with gradio.Tab('Settings'):
                    thin_client_cbx = gradio.Checkbox(
                        label='Thin-client mode',
                        info="Only generate images remotely (no image previews yet)",
                        value=self.world.thin_client_mode
                    )
                    job_timeout = gradio.Number(
                        label='Job timeout', value=self.world.job_timeout,
                        info="Seconds until a worker is considered too slow to be assigned an"
                             " equal share of the total request. Longer than 2 seconds is recommended"
                    )

                    complement_production = gradio.Checkbox(
                        label='Complement production',
                        info='Prevents under-utilization by requesting additional images when possible',
                        value=self.world.complement_production
                    )

                    # reduces image quality the more the sample-count must be reduced
                    # good for mixed setups where each worker may not be around the same speed
                    step_scaling = gradio.Checkbox(
                        label='Step scaling',
                        info='Prevents under-utilization via sample reduction in order to meet time constraints',
                        value=self.world.step_scaling
                    )

                    save_btn = gradio.Button(value='Update')
                    save_btn.click(fn=self.save_btn, inputs=[thin_client_cbx, job_timeout, complement_production, step_scaling])
                    components += [thin_client_cbx, job_timeout, complement_production, step_scaling, save_btn]

                with gradio.Tab('Help'):
                    gradio.Markdown(
                        """
                        - [Discord Server 🤝](https://discord.gg/Jpc8wnftd4)
                        - [Github Repository </>](https://github.com/papuSpartan/stable-diffusion-webui-distributed)
                        """
                    )

            # prevent wui from overriding any values
            for component in components:
                setattr(component, 'do_not_save_to_config', True)  # ui_loadsave.py apply_field()
            return components
