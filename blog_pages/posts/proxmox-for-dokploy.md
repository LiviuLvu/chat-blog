---
title: "Running Proxmox and Dokploy for my home lab"
date: "2026-01-01"
summary: "Proxmox is my new virtualization platform of choice for running containerized workloads at home. After several detours through configuration rabbit holes, here's what I learned setting up LXC containers for Dokploy and other services."
# description: ""
tags: ["Home lab", "Proxmox", "Dokploy"]
author: "Liviu Iancu"
weight: 1
series: ["Devops"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---

Proxmox is my new virtualization platform of choice for running containerized workloads at home. After several detours through configuration rabbit holes, here's what I learned setting up LXC containers for Dokploy and other services.

## Choosing the Right OS for Dokploy

Debian 12 didn't cooperate. Missing curl dependencies and repository connection issues made me switch quickly. Ubuntu 14.4 worked smoothly—curl installed cleanly and Dokploy deployed with a single command.

My initial LXC container specs were generous, probably too generous:
- 8 cores
- 8GB memory
- 64GB storage
- 512MB swap

I'll scale these down after monitoring actual usage.

## The IP Address problem

Here's where I lost hours: Proxmox saves the initial DHCP-assigned IP as static in `/etc/network/interfaces`. When I changed the container to a different static IP, the Opnsense router kept showing the old address.

The fix was editing Proxmox's network config:
```bash
nano /etc/network/interfaces
# Change this line from static to dhcp
iface vmbr0 inet static
```

I repeated this exact mistake with Docker Swarm. Changing the host IP after swarm initialization broke everything. A forced `docker swarm leave -f` and reinit disconnected Dokploy completely. I ended up reinstalling Dokploy rather than debugging further.

Pro tip: decide your IP addressing scheme before deploying Dokploy. Save yourself the headache.

## Essential Proxmox Configuration

After installation:
1. Disable enterprise repositories
2. Add no-subscription repos
3. Update package lists
4. Reboot (unfortunately necessary)

Keep unprivileged containers enabled (the default). This prevents containers from accessing the entire host system.

## Finding Container Ports

New applications deployed through Dokploy are accessible via `http://<host-ip>:<port>`. The port mapping isn't obvious from Docker files.

Navigate to: Dashboard → Docker tab → Row actions (three dots) → View config → Search for "HostPort"

That three-dot menu hides surprisingly well, if you have a small screen .

## USB Passthrough for Zigbee
If you are using Homeassistant, getting home automation hardware connected is straightforward:
```bash
lsusb  # Check what's available
ls /dev/serial/by-id/  # List USB devices
```

In Proxmox check: Container → Resources → Add → Device passthrough
My Zigbee dongle path for example is:
```
/dev/serial/by-id/usb-dresden_elektronik_ingenieurtechnik_GmbH_ConBee_II_DE2478649-if00
```
If the usb is recognized, you need to add it to your LXC container (in this case Dokploy) from the proxmox interface.

Some changes required in the Homeassistant compose file inside Dokploy: 
- restart: in case the host machine / dockploy is restarted
- devices: manual mapping of usb from dokploy LXC container to HA container
- group_add: adds Homeassistant user to root group, for usb access permission

```
services:
  homeassistant:
    image: ghcr.io/home-assistant/home-assistant:stable
    restart: unless-stopped
    devices:
      - /dev/serial/by-id/<usb_file_name>:/dev/serial/by-id/<usb_file_name>
    group_add:
      - "0"

```

## LXC Config Files

Container configurations live at `/etc/pve/lxc/`. Not sure when I'll need direct access to these, but it's good to know in case I want to manually test a setting.

## Helper Scripts and Community Templates

The [Proxmox Community Scripts](https://community-scripts.github.io/ProxmoxVE/scripts) project offers quick setups for various services. I prefer following official installation instructions, though. For Dokploy, I created a basic Ubuntu LXC, installed curl, and ran their one-line installer.

Always review scripts before running them. At minimum, have an AI analyze it if you're short on time.

## Getting Back to Building Apps

Facepalm moment: I spent hours troubleshooting that IP change when I could have been building applications. All this because I changed from dynamic to static IP mid-deployment. Next time I'll do the planning upfront.

Docker Swarm through Dokploy makes container management simple. I deployed Convex DB with its dashboard in minutes. The built-in metrics need zero configuration—watching resource usage without manual Prometheus setups feels refreshing.

Portainer runs alongside Dokploy for additional statistics and container management options. Both installed in under five minutes.

Proxmox rewards experimentation. Each configuration teaches something new about networking, containers, or Linux systems. My next challenge: backup strategies and network storage.

What's your home lab setup? Share your Proxmox experiences or container orchestration choices. I'm curious how others approach the storage puzzle.