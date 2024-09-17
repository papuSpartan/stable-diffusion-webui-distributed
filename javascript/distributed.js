
function confirm_restart_workers(_) {
    return confirm('Restart remote workers?')
}

// live updates
function update() {
    try {
        let refresh_button = document.getElementById('distributed-refresh-status')
        refresh_button.click()
    } catch (e) {
        if (!(e instanceof TypeError)) {
            throw e
        }
    }
}
setInterval(update, 1500)