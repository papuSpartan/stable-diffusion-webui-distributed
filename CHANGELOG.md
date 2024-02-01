# Change Log
Formatting: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [2.0.2] - 2024-1-31

### Fixed
- Potential hang after first request since startup
- Extension parity warnings

## [2.0.1] - 2024-1-25

### Fixed
- Crashing if not at debug log level

## [2.0.0] - 2024-1-24

### Added
- A changelog
- Faster model loading by hooking the model dropdown and immediately sending sync requests
- Popups in UI to lessen the need to check the status tab 
- A debug option for resetting error correction at runtime
- Experimental "Pixel Cap" setting under worker config UI tab for limiting pixels allocated to a given worker

### Changed
- From a selectable script to an **AlwaysOn** style extension
- Thin-client mode will lack function for now
- Renamed main extension file from `extension.py` -> `distributed.py` to help with config grouping
- The log file (`distributed.log`) will max out at a size of 10MB and rotate over
- All relevant UI components are returned by ui() which has implications for API usage

### Fixed
- Connection check ping not always being sent on startup
- Old issue where benchmarking could become inaccurate due to models not being loaded on every machine [#11](https://github.com/papuSpartan/stable-diffusion-webui-distributed/issues/11)
- Worker randomly disconnecting when under high load due to handling a previous request

### Removed
- Certain superfluous warnings in logs related to third party extensions