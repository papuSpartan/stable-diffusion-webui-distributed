import os
from pathlib import Path
from inspect import getsourcefile
from os.path import abspath

def preload(parser):
	parser.add_argument(
		"--distributed-remotes",
		nargs="+",
		help="Enter n pairs of sockets",
		type=lambda t: t.split(":")
	)

	parser.add_argument(
		"--distributed-skip-verify-remotes",
		help="Disable verification of remote worker TLS certificates",
		action="store_true"
	)

	parser.add_argument(
		"--distributed-remotes-autosave",
		help="Enable auto-saving of remote worker generations",
		action="store_true"
	)

	parser.add_argument(
		"--distributed-debug",
		help="Enable debug information",
		action="store_true"
	)
	webui_root_path = Path(abspath(getsourcefile(lambda: 0))).parent.parent.parent
	config_path = webui_root_path.joinpath('distributed-config.json')
	# add config file
	parser.add_argument(
		"--distributed-config",
		help="config file to load / save, default: distributed-config.json",
		default=config_path
	)
