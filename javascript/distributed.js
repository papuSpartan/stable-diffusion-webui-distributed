
function confirm_restart_workers(_) {
    return confirm('Restart remote workers?')
}

// live updates for extension status tab
function update() {
    try {
        get_uiCurrentTabContent().querySelectorAll('#distributed-refresh-status')[0].click()
    } catch (e) {
        if (!(e instanceof TypeError)) {
            throw e
        }
        console.log('distributed ext: sdwui page not yet loaded... waiting')
    }
}
setInterval(update, 1500)