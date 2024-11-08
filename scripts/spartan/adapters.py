from typing_extensions import override, Tuple
from scripts.spartan.shared import logger
from scripts.spartan.control_net import pack_control_net

class Adapter(object):
	def __init__(self):
		self.script = None

	def early(self, p, world, script, *args):
		self.script = script

	def less_early(self, p, world, payload, *args):
		pass

	def script_args(self, p) -> Tuple:
		return p.script_args[self.script.args_from:self.script.args_to]

class DynamicPromptsAdapter(Adapter):
	def __init__(self):
		super().__init__()
		self.title = "Dynamic Prompts"

	def early(self, p, world, script, *args):
		super().early(p, world, script)
		
		self.script = script
		logger.debug("finding callback")

		script_process_cbs = p.scripts.callback_map['script_process'][1]

		for i, callback in enumerate(script_process_cbs):
			if callback.callback.name == self.title.lower():
				logger.debug(f"found callback")

				# prevent double exec
				script_process_cbs.remove(callback)
		else:
			logger.debug(f"already hooked dynamic prompts")

	# right before payload is cloned into a seperate instance for each job
	def less_early(self, p, world, payload, *args):
		logger.debug("running dynprompts early")

		# dynamic clobbers the actual p even if we pass a different object
		p.all_prompts.clear()
		for i in range(world.num_requested()):
			p.all_prompts.append(p.prompt)
		# dynprompts_args = p.script_args[self.script.args_from:self.script.args_to]
		# dynamic prompts overrides p.all_prompts if it doesn't match the batch size. we should be able to set batch-
		# size earlier than in the past when we were running as a selectable script since we are overriding local generation
		# we can probably just set this earlier since
		p.batch_size = world.num_requested()
		self.script.process(p, *self.script_args(p))
		# TODO this shouldn't need to be done twice (only doing now for dyn prompts)
		payload['all_prompts'] = p.all_prompts

class ControlNetAdapter(Adapter):
	def __init__(self, *args):
		super().__init__()
		self.title = "ControlNet"

	def early(self, p, world, script, packed_script_args, *args):
		super().early(p, world, script)

		# grab all controlnet units
		cn_units = []
		for cn_arg in self.script_args(p):
			if "ControlNetUnit" in type(cn_arg).__name__:
				cn_units.append(cn_arg)
		logger.debug(f"Detected {len(cn_units)} controlnet unit(s)")

		# get api formatted controlnet
		packed_script_args.append(pack_control_net(cn_units))

class ADetailerAdapter(Adapter):
	def __init__(self, *args):
		super().__init__()
		self.title = "ADetailer"

	def early(self, p, world, script, *args):
		super().early(p, world, script)
		adetailer_args = self.script_args(p)

		# InputAccordion main toggle, skip img2img toggle
		if adetailer_args[0] and adetailer_args[1]:
			logger.debug(f"adetailer is skipping img2img, returning control to wui")
			return

adapters = [ControlNetAdapter(), ADetailerAdapter(), DynamicPromptsAdapter()]
