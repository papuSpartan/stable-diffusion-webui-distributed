import launch

if not launch.is_installed("rich"):
    launch.run_pip("install rich", "requirements for distributed")
