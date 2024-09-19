
function confirm_restart_workers(_) {
    return confirm('Restart remote workers?')
}

// live updates
function update() {
    try {
        let currentTab = get_uiCurrentTabContent()
        let buttons = document.querySelectorAll('#distributed-refresh-status')
        for(let i = 0; i < buttons.length; i++) {
            if(currentTab.contains(buttons[i])) {
                buttons[i].click()
                break
            }
        }
    } catch (e) {
        if (!(e instanceof TypeError)) {
            throw e
        }
    }
}
setInterval(update, 1500)