# stable-diffusion-webui-distributed
This extension enables you to chain multiple webui instances together for txt2img and img2img generation tasks.

There is an emphasis on minimizing the perceived latency/lag of large batch jobs in the **main** sdwui instance.

![alt text](doc/sdwui_distributed.drawio.png)\
*Diagram showing Master/slave architecture of the extension*

**Contributions and feedback are much appreciated!**

[![](https://dcbadge.vercel.app/api/server/Jpc8wnftd4)](https://discord.gg/Jpc8wnftd4)

## Installation

On the master instance:
- Go to the extensions tab, and swap to the "available" sub-tab. Then, search "Distributed", and hit install on this extension.

On each slave instance:
- enable the api by passing `--api` and ensure it is listening by using `--listen`
- ensure all of the models, scripts, and whatever else you think you might request from them is present\
Ie. if you're using sd-1.5 on the controlling instance, then the sd-1.5 model should also be present on each slave instance. Otherwise, the remote will fallback to some other model that **is** present.

*if you want to easily sync models between your nodes, you might want to use something like [rclone](https://rclone.org/)*

### Tips
- This is not an **always on** script, you have to select it in the script dropdown of the tab you are in.
- If benchmarking fails, try hitting the **Redo benchmark** button under the script's **Util** tab.
- If any remote is taking far too long to returns its share of the batch, you can hit the **Interrupt** button in the **Util** tab.

### Command-line arguments

**--distributed-remotes** Enter n pairs of sockets corresponding to remote workers in the form `name:address:port` (deprecated)\
**--distributed-skip-verify-remotes** Disable verification of remote worker TLS certificates (useful for if you are using self-signed certs like with auto tls-https)\
**--distributed-remotes-autosave** Enable auto-saving of remote worker generations\
**--distributed-debug** Enable debug information
