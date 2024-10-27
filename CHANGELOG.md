# Change Log
Formatting: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [2.3.0] - 2024-10-26

## Added
- Compatibility for some extensions which mostly only do postprocessing (e.g. Adetailer)
- Separate toggle state for img2img tab so txt2img can be enabled and t2i disabled or vice versa

## Changed
- Status tab will now automatically refresh
- Main toggle is now in the form of an InputAccordion

## Fixed
- An issue affecting controlnet and inpainting
- Toggle state sometimes desyncing when the page was refreshed

## [2.2.2] - 2024-8-30

### Fixed
- Unavailable state sometimes being ignored

## [2.2.1] - 2024-5-16

### Fixed
- Grid generation regression
- Model propagation error handling

## [2.2.0] - 2024-5-11

### Added
- Toggle for allowing automatic step scaling which can increase overall utilization

### Changed
- Adding workers which have the same socket definition as master will no longer be allowed and an error will show #28 
- Workers in an invalid state should no longer be benchmarked
- The worker port under worker config will now default to 7860 to prevent mishaps
- Config should once again only be loaded once per session startup
- A warning will be shown when trying to use the user script button but no script exists

### Fixed
- Thin-client mode
- Some problems with sdwui forge branch
- Certificate verification setting sometimes not saving
- Master being assigned no work stopping generation (same problem as thin-client)

### Removed
- Adding workers using deprecated cmdline argument

## [2.1.0] - 2024-3-03

### Added
- Ability to disable complementary/"bonus" image production under settings tab
- Utility for quickly restarting all remote machines under utils tab
- Prefix in logging to visually separate from other extensions or output

### Changed
- Improved UI
- Complementary production is now highlighted in the job distribution summary

### Fixed
- Problem preventing automatic benchmarking
- Gradio deprecation warnings

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
