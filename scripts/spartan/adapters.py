from PIL.Image import Image
from typing_extensions import override, Tuple
from scripts.spartan.shared import logger
from scripts.spartan.control_net import pack_control_net

class Adapter(object):
	def __init__(self):
		self.script = None

	def early(self, p, world, script, *args) -> bool:
		"""make changes before any worker request objects are created. return True to cede control back to webui"""

		self.script = script
		return False

	def late(self, p, world, payload, *args):
		"""make changes after the worker request object has been created and workloads have been manipulated"""
		# payload['alwayson_scripts'] guaranteed to exist, but may not be populated
		pass

	def script_args(self, p) -> Tuple:
		return p.script_args[self.script.args_from:self.script.args_to]

class DynamicPromptsAdapter(Adapter):
	def __init__(self):
		super().__init__()
		self.title = "Dynamic Prompts"

	def early(self, p, world, script, *args):
		super().early(p, world, script)

		# dynamic will run twice, but because of load order our call overwrites the original
		# logger.debug("finding callback")
		# script_process_cbs = p.scripts.callback_map['script_process'][1]

		# for i, callback in enumerate(script_process_cbs):
		# 	if callback.callback.name == self.title.lower():
		# 		logger.debug(f"found callback")

		# 		# prevent double exec
		# 		script_process_cbs.remove(callback)
		# else:
		# 	logger.debug(f"already hooked dynamic prompts")

	# right before payload is cloned into a seperate instance for each job
	def late(self, p, world, payload, *args):

		# dynamic clobbers the actual p even if we pass a different object
		if len(p.prompt) > 1:
			p.all_prompts.clear()
			for i in range(world.num_requested()):
				p.all_prompts.append(p.prompt)
		elif len(p.negative_prompt) > 1:
			p.all_negative_prompts.clear()
			for i in range(world.num_requested()):
				p.all_negative_prompts.append(p.negative_prompt)
		else:
			return

		# dynamic prompts overrides p.all_prompts if it doesn't match the batch size
		temp = p.batch_size
		p.batch_size = world.num_requested()
		self.script.process(p, *self.script_args(p))
		p.batch_size = temp
		payload['all_prompts'] = p.all_prompts
		payload['all_negative_prompts'] = p.all_negative_prompts

class ControlNetAdapter(Adapter):
	def __init__(self, *args):
		super().__init__()
		self.title = "ControlNet"

	def late(self, p, world, payload, *args):
		# grab all controlnet units
		cn_units = []
		for cn_arg in self.script_args(p):
			if "ControlNetUnit" in type(cn_arg).__name__:
				cn_units.append(cn_arg)
		logger.debug(f"Detected {len(cn_units)} controlnet unit(s)")

		# get api formatted controlnet
		payload['alwayson_scripts'].update(pack_control_net(cn_units))

class ADetailerAdapter(Adapter):
	def __init__(self, *args):
		super().__init__()
		self.title = "ADetailer"

	def early(self, p, world, script, *args):
		super().early(p, world, script)
		adetailer_args = self.script_args(p)

		# InputAccordion main toggle, skip img2img toggle
		if adetailer_args[0] and adetailer_args[1]:
			return True

	def late(self, p, world, payload, *args):
		payload['_ad_orig'] = None # unserializable


class GenericAdapter(Adapter):
	def __init__(self, *args):
		super().__init__()
		self.packed_script_args = []  # list of api formatted per-script argument objects
		# { "script_name": { "args": ["value1", "value2", ...] }

	def early(self, p, world, script, *args):
		title = script.title()
		# https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/12#issuecomment-1480382514
		args_script_pack = {title: {"args": []}}
		for arg in p.script_args[script.args_from:script.args_to]:
			args_script_pack[title]["args"].append(arg)
		self.packed_script_args.append(args_script_pack)

	def late(self, p, world, payload, *args):
		for packed in self.packed_script_args:
			payload['alwayson_scripts'].update(packed)


		if payload.get('init_images_original_md') is not None: # multidiffusion
			payload['init_images_original_md'] = None

		# for key in payload:
		# 	contains_dict = any(isinstance(payload[key], Image) for item in payload)
		# 	if isinstance(payload[key], Image):
		# 		logger.warning(f"will not serialize PIL image in key '{key}'")
		# 		del payload[key]



adapters = [ControlNetAdapter(), ADetailerAdapter(), DynamicPromptsAdapter()]
