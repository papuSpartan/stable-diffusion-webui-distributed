# stable-diffusion-webui-distributed
This extension allows one to form a distributed compute system by linking up multiple instances of stable-diffusion-webui to one master instance. In practice, this means being able to tell a bunch of machines to generate images at the same time and have them sent right back to your controlling machine. The main goal is to maximize concurrency and minimize the latency of large-scale image generation in the main sdwui instance.

Pros:
- should be able to enslave any sdwui running in api mode
- can be disabled at any time on the main instance just like a normal script
- can be used with sdwui api instances setup to use [HTTPS](https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https)
- should work with any combination of models, vae, and lora

Cons:
- currently only suports txt2img
- not a low-level implementation. Meaning, it will not "combine" your VRAM and compute a singular huge image as some might conclude at first glance
- currently you are responsible for each remote worker having the models and etc. that you request to use from the master instance
- currently requires a patch (only to the master sdwui instance)
- currently does not predict out of memory exceptions (first on priority)

**Pull requests and bug reports are much appreciated**

## How it works (currently)
Say you want to generate 12 images and you hit the generate button on the master instance:
1. If there is no workers.json file, it will benchmark every machine(worker) and save that information to workers.json
2. Assume we have 3 workers, with each worker measured to run at ~20ipm. Images will be split equally among them.
3. The master instance (the UI you are looking at) will begin generating its portion of the images(4) like it would if you had set the batch_size slider to 4 normally
4. Once the 4 images are done and the image viewer appears, the extension will start adding all of the images received from the remote machines to the gallery.
5. Profit?

That was the simple case though, step 2 gets much more complicated if the machines' compute speeds and/or memory sizes are much different. For example, a setup which utilizes 3 distinct workers that operate at 5, 15, and 20 ipm each would have the following job assignment if the master instance requests 12 images:
```
After job optimization, job layout is the following:
worker 'master' - 8 images
worker 'laptop' - 12 images
worker 'argon' - 3 images
```

The reason it works like this is the following:
- 'laptop' is the fastest real-time worker at 20 ipm so it (initially) gets dealt an equal share of 4 images
- both of the other workers are considered 'complementary' workers because they cannot keep up with 'laptop' **enough***
- because of my goal for this extension, both 'complementary' workers will calculate how much, in addition, they **can** make in the time that 'laptop' will take to make the main 12.


## Installation
**DISCLAIMER - I do NOT recommend casual users use this yet unless you are okay with the possibility of severe instability**

On the master instance:
- install the [necessary patch](https://gist.github.com/papuSpartan/e300f5a7030b6179f7e73dd6f75891fb)
- Go to the "Extensions" tab and then swap to the "Install from URL" tab
- Paste https://github.com/papuSpartan/stable-diffusion-webui-distributed.git into "URL for extension's git repository" and click install
- Ensure that you have setup your `COMMANDLINE_ARGS` include the necessary info on your remote machines. Ex:
```
set COMMANDLINE_ARGS=--distributed-remotes laptop:192.168.1.3:7860 argon:fake.local:7860 --distributed-skip-verify-remotes --distributed-remotes-autosave
```

On each slave instance:
- enable the api by passing `--api` and ensure it is listening by using `--listen`
- ensure all of the models, scripts, and whatever else you think you might request from the master is present\ 

*if you're wanting to sync a huge amount of nodes you might want to use something like rsync or winscp*

# Usage Notes
- If benchmarking fails, just delete the workers.json file generated in the extension folder and try again.
- You need to have all of the workers you plan to use connected when benchmarking for things to work properly (will be fixed later).

### Command-line arguments

**--distributed-remotes** Enter n pairs of sockets corresponding to remote workers in the form `name:address:port`\
**--distributed-skip-verify-remotes** Disable verification of remote worker TLS certificates (useful for if you are using self-signed certs like with auto tls-https)\
**--distributed-remotes-autosave** Enable auto-saving of remote worker generations
