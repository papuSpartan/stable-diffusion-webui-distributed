# stable-diffusion-webui-distributed
This extension enables you to chain multiple webui instances together for txt2img and img2img generation tasks.

*For those with **multi-gpu** setups, **yes** this can be used for generation across all of those devices.*

The main goal is minimizing the lag of (high batch size) requests from the **main** sdwui instance.

![alt text](doc/sdwui_distributed.drawio.png)\
*Diagram shows Master/slave architecture of the extension*

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
- If benchmarking fails, try hitting the **Redo benchmark** button under the script's **Util** tab.
- If any remote is taking far too long to returns its share of the batch, you can hit the **Interrupt** button in the **Util** tab.
- If you think that a worker is being under-utilized, you can adjust the job timeout setting to be higher. However, doing this may be suboptimal in cases where the "slow" machine is **actually** really slow. Alternatively, you may just need to do a re-benchmark or manually edit the config.

### Command-line arguments

**--distributed-skip-verify-remotes** Disable verification of remote worker TLS certificates (useful for if you are using self-signed certs like with [auto tls-https](https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https))\
**--distributed-remotes-autosave** Enable auto-saving of remote worker generations\
**--distributed-debug** Enable debug information
